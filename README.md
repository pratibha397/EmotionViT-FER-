# EmotionViT-FER: Facial Emotion Recognition with Vision Transformers

Facial emotion recognition using Vision Transformers with custom region attention mechanism. Achieves 84.88% validation accuracy on FER2013+ dataset.

## 📊 Dataset

### FER2013+ Dataset
- **Source**: Extended version of FER2013 with additional samples and corrected labels -> https://www.kaggle.com/datasets/subhaditya/fer2013plus
- **Classes**: 8 emotions (anger, contempt, disgust, fear, happiness, neutral, sadness, surprise)
- **Size**: 
  - Training: 28,386 samples
  - Testing: 7,099 samples
- **Format**: 48x48 pixel grayscale images (converted to RGB for ViT)
- **Distribution**:   anger: 3,995 samples
                      contempt: 593 samples
                      disgust: 436 samples
                      fear: 4,097 samples
                      happiness: 7,215 samples
                      neutral: 4,965 samples
                      sadness: 4,830 samples
                      surprise: 3,171 samples

  

## 🏗️ Model Architecture

### EmotionTransformer Architecture
1. **Backbone**: Pre-trained ViT-Base (16x16 patches, 224x224 input)
2. **Region Attention Module**:
   - Learns attention weights for 8 facial regions
   - Produces emotion-specific feature maps
3. **Emotion-Specific Heads**:
   - 8 parallel branches (one per emotion)
   - Each: Linear(768) → ReLU → Dropout(0.3) → Linear(1)
4. **Fusion Layer**:
   - Combines global features with emotion-specific predictions
   - Final classification with 128-unit hidden layer

### Key Innovations
- **Region-based attention**: Focuses on facial regions relevant to each emotion
- **Multi-head emotion processing**: Separate pathways for different emotions
- **Feature fusion**: Combines global and emotion-specific representations

## 📈 Results

### Performance Metrics
| Metric        | Value   |
|---------------|---------|
| Validation Acc| 84.88%  |
| Training Acc  | 96.08%  |
| Test Acc      | 83.45%  |
| Best Epoch    | 15      |


### Per-Class Performance
| Emotion    | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| anger      | 0.82      | 0.79   | 0.80     |
| contempt   | 0.71      | 0.68   | 0.69     |
| disgust    | 0.76      | 0.74   | 0.75     |
| fear       | 0.80      | 0.78   | 0.79     |
| happiness  | 0.95      | 0.96   | 0.95     |
| neutral    | 0.85      | 0.87   | 0.86     |
| sadness    | 0.83      | 0.85   | 0.84     |
| surprise   | 0.88      | 0.86   | 0.87     |

## 🚀 Setup & Usage

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

# Clone repository
git clone https://github.com/pratibha397/EmotionViT-FER-.git
cd EmotionViT-FER

# Install dependencies
pip install -r requirements.txt
