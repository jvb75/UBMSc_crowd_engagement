from werkzeug.exceptions import RequestEntityTooLarge
# app.py (updated with proper file upload handling)
from flask import Flask, render_template, request, jsonify, send_file
import torch
from model_architecture import CNNLSTM
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import json
import yaml
from pathlib import Path
import tempfile
import io
import base64
from datetime import datetime
import os
import traceback

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Global variables
config = None
pipeline = None
device_info = "Unknown"

# Load configuration
def load_config():
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Create default config if file doesn't exist
        default_config = {
            'models': {
                'event': 'models/event_cnnlstm_best.pt',
                'engagement': 'models/engagement_cnnlstm_best.pt',
                'environment': 'models/env_cnnlstm_best.pt',
                'density': None
            },
            'audio': {
                'sr': 22050,
                'max_duration_s': 15,
                'normalize': True
            },
            'features': {
                'type': 'mel',
                'n_mels': 128,
                'n_fft': 2048,
                'hop_length': 512,
                't_max': 427
            }
        }
        return default_config

class AudioInferencePipeline:
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.models = {}
        self.meta = self._load_meta()
        self._load_models()
    
    def _get_device(self):
        device_config = self.config.get('device', 'auto')
        if device_config == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device_config)
    
    def _load_meta(self):
        meta_path = self.config.get('meta_path', 'models/cnn_lstm_multitask_meta.json')
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Metadata file {meta_path} not found. Using default values.")
            # Return default metadata structure
            return {
                'sr': 22050,
                'features': {
                    'type': 'mel',
                    'n_mels': 128,
                    'n_fft': 2048,
                    'hop_length': 512,
                    't_max': 427
                },
                'preprocessing': {
                    'mu': [-48.86],
                    'sigma': [10.89]
                },
                'label_maps': {
                    'event': {'0': 'music', '1': 'speech', '2': 'conversation', '3': 'crowd noise'},
                    'engagement': {'0': 'low', '1': 'medium', '2': 'high'},
                    'environment': {'0': 'indoor', '1': 'outdoor'}
                }
            }
    

    def _load_models(self):
        model_paths = {
            'event': self.config['models'].get('event'),
            'engagement': self.config['models'].get('engagement'),
            'environment': self.config['models'].get('environment'),
            'density': self.config['models'].get('density')
        }
        # Get input shape from metadata
        input_shape = (
            self.meta['features']['t_max'],
            self.meta['features']['n_mels']
        )
        for head, path in model_paths.items():
            if path and os.path.exists(path):
                try:
                    # Load state dict
                    state_dict = torch.load(path, map_location=self.device)
                    # Determine number of classes for this head
                    if head == 'density':
                        num_classes = self.meta.get('density_classes', 9) - 1
                    else:
                        num_classes = len(self.meta['label_maps'].get(head, {}))
                    # Create model architecture
                    model = CNNLSTM(input_shape, num_classes, self.meta.get('model_architecture', {}))
                    # Load weights
                    model.load_state_dict(state_dict)
                    model.to(self.device)
                    model.eval()
                    self.models[head] = model
                    print(f"Loaded {head} model from {path}")
                except Exception as e:
                    print(f"Warning: Could not load {head} model from {path}: {str(e)}")
                    print(traceback.format_exc())
            else:
                print(f"Warning: Model path for {head} not found or not specified: {path}")

    def preprocess_audio(self, audio_path):
        try:
            # Load audio
            sr = self.meta.get('sr', self.config['audio']['sr'])
            audio, _ = librosa.load(audio_path, sr=sr, mono=True)
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Extract features
            if self.meta['features']['type'] == 'mel':
                mel_spec = librosa.feature.melspectrogram(
                    y=audio,
                    sr=sr,
                    n_mels=self.meta['features']['n_mels'],
                    n_fft=self.meta['features']['n_fft'],
                    hop_length=self.meta['features']['hop_length'],
                )
                features = librosa.power_to_db(mel_spec, ref=np.max)
                features = features.T  # (T, F)
            
            # Pad/truncate
            T_max = self.meta['features']['t_max']
            if features.shape[0] < T_max:
                pad_width = ((0, T_max - features.shape[0]), (0, 0))
                features = np.pad(features, pad_width, mode='constant')
            else:
                features = features[:T_max]
            
            # Standardize
            mu = np.array(self.meta['preprocessing']['mu'])
            sigma = np.array(self.meta['preprocessing']['sigma'])
            features = (features - mu) / sigma
            
            # Convert to tensor
            features = torch.FloatTensor(features).unsqueeze(0)  # (1, T, F)
            
            return features
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise
    
    def predict(self, audio_path):
        try:
            features = self.preprocess_audio(audio_path)
            features = features.to(self.device)
            
            result = {}
            
            with torch.no_grad():
                for head, model in self.models.items():
                    outputs = model(features)
                    
                    if head == 'density' and outputs.shape[1] > 1:  # Ordinal regression
                        predictions = (torch.sigmoid(outputs) > 0.5).sum(dim=1)
                        probabilities = torch.zeros(outputs.shape[0], self.meta.get('density_classes', 9))
                        for i, pred in enumerate(predictions):
                            probabilities[i, pred.item()] = 1.0
                    else:
                        probabilities = torch.softmax(outputs, dim=1)
                    
                    probs = probabilities.cpu().numpy()[0]
                    class_idx = np.argmax(probs)
                    
                    # Map to class labels
                    id2label = self.meta['label_maps'].get(head, {})
                    top_class = id2label.get(str(class_idx), f"class_{class_idx}")
                    confidence = probs[class_idx]
                    
                    # Get all class probabilities
                    class_probs = {}
                    for idx, prob in enumerate(probs):
                        class_probs[id2label.get(str(idx), f"class_{idx}")] = float(prob)
                    
                    result[head] = {
                        'top_class': top_class,
                        'confidence': confidence,
                        'probabilities': class_probs
                    }
            
            return result
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {'error': str(e)}

def plot_waveform(audio_path, sr):
    try:
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        
        fig, ax = plt.subplots(figsize=(10, 3))
        time = np.arange(len(audio)) / sr
        ax.plot(time, audio)
        ax.set_title("Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        print(f"Error creating waveform: {str(e)}")
        return ""

def plot_spectrogram(audio_path, config):
    try:
        audio, sr = librosa.load(audio_path, config['audio']['sr'])
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=config['features']['n_mels'],
            n_fft=config['features']['n_fft'],
            hop_length=config['features']['hop_length']
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            mel_spec_db,
            x_axis='time',
            y_axis='mel',
            sr=sr,
            ax=ax
        )
        ax.set_title("Mel Spectrogram")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        print(f"Error creating spectrogram: {str(e)}")
        return ""

# Initialize the application
def init_app():
    global config, pipeline, device_info
    
    try:
        config = load_config()
        pipeline = AudioInferencePipeline(config)
        device_info = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"Inference pipeline initialized successfully on {device_info}")
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        print(traceback.format_exc())
        pipeline = None
        device_info = "Error"

# Initialize when the app starts
init_app()

@app.route('/')
def index():
    return render_template('index.html', device_info=device_info)


@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Check total size before processing
        total_size = 0
        for file in files:
            if file.filename != '':
                file.seek(0, 2)  # Seek to end to get file size
                total_size += file.tell()
                file.seek(0)  # Reset file pointer
                
        # Limit total upload size to 200MB
        if total_size > 200 * 1024 * 1024:
            return jsonify({'error': 'Total upload size exceeds 200MB limit'}), 413
        
        results = []
        for file in files:
            if file.filename == '':
                continue
                
            try:
                # Check individual file size
                file.seek(0, 2)
                file_size = file.tell()
                file.seek(0)
                
                if file_size > 100 * 1024 * 1024:  # 100MB per file
                    results.append({
                        'filename': file.filename,
                        'error': f'File too large ({file_size/(1024*1024):.1f}MB). Maximum 100MB per file.'
                    })
                    continue
                
                # Check file extension
                allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in allowed_extensions:
                    results.append({
                        'filename': file.filename,
                        'error': f'Unsupported file type: {file_ext}. Supported: {", ".join(allowed_extensions)}'
                    })
                    continue
                
                # Save temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                    file.save(tmp_file.name)
                    audio_path = tmp_file.name
                
                # Generate previews (skip for very large files to save time)
                waveform_img = ""
                spectrogram_img = ""
                if file_size < 10 * 1024 * 1024:  # Only generate previews for files < 10MB
                    try:
                        waveform_img = plot_waveform(audio_path, config['audio']['sr'])
                        spectrogram_img = plot_spectrogram(audio_path, config)
                    except Exception as e:
                        print(f"Warning: Could not generate preview for {file.filename}: {e}")
                
                # Run inference
                if pipeline:
                    result = pipeline.predict(audio_path)
                else:
                    result = {'error': 'Pipeline not initialized'}
                
                # Clean up
                try:
                    os.unlink(audio_path)
                except:
                    pass
                
                # Convert all numpy.float32 in result to float for JSON serialization
                def convert_floats(obj):
                    if isinstance(obj, dict):
                        return {k: convert_floats(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_floats(i) for i in obj]
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    return obj

                results.append({
                    'filename': file.filename,
                    'waveform': waveform_img,
                    'spectrogram': spectrogram_img,
                    'predictions': convert_floats(result),
                    'file_size': f"{file_size/(1024*1024):.1f}MB"
                })
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': f"Processing error: {str(e)}"
                })
        
        return jsonify({'results': results})
        
    except RequestEntityTooLarge:
        return jsonify({'error': 'Total upload size exceeds capacity limit'}), 413
    except Exception as e:
        print(f"Upload error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/download', methods=['POST'])
def download_results():
    try:
        data = request.get_json()
        if not data or 'results' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Prepare CSV data
        csv_data = []
        for result in data['results']:
            if 'error' in result:
                continue
                
            row = {
                'filename': result['filename'],
                'file_size': result.get('file_size', 'N/A')
            }
            
            if 'predictions' in result:
                for head in ['event', 'engagement', 'environment', 'density']:
                    if head in result['predictions']:
                        pred = result['predictions'][head]
                        row[f'{head}_prediction'] = pred['top_class']
                        row[f'{head}_confidence'] = f"{pred['confidence']:.3f}"
                        for cls, prob in pred['probabilities'].items():
                            row[f'{head}_{cls}'] = f"{prob:.3f}"
                    else:
                        row[f'{head}_prediction'] = 'N/A'
                        row[f'{head}_confidence'] = 'N/A'
            csv_data.append(row)
        
        # Create DataFrame and CSV
        df = pd.DataFrame(csv_data)
        
        # Reorder columns for better readability
        column_order = ['filename', 'file_size']
        for head in ['event', 'engagement', 'environment', 'density']:
            column_order.extend([f'{head}_prediction', f'{head}_confidence'])
            # Add probability columns for each class
            if csv_data and any(f'{head}_' in key for key in csv_data[0].keys()):
                prob_columns = [k for k in csv_data[0].keys() if k.startswith(f'{head}_') and not k.endswith(('prediction', 'confidence'))]
                column_order.extend(sorted(prob_columns))
        
        df = df.reindex(columns=column_order)
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'audio_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        
    except Exception as e:
        print(f"Download error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

# app.py - add this error handler
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
