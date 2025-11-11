# GenAI-For-Predictive-Maintenance-In-EV
# Predictive Maintenance Dashboard

A comprehensive web application for predictive maintenance using machine learning models to detect anomalies and predict remaining useful life (RUL) of industrial equipment.

## Features

### 🔧 Core Functionality
- **Anomaly Detection**: VAE-LSTM model for real-time anomaly detection in time series data
- **RUL Prediction**: XGBoost regression model for remaining useful life estimation
- **Interactive Dashboard**: Real-time monitoring and visualization of equipment health
- **Data Processing**: Automated preprocessing pipeline with outlier handling and feature engineering

### 📊 Web Application (Streamlit)
- **Dashboard**: Project summary, model status, and dataset overview
- **Data Explorer**: Upload CSV/ZIP files, preview data, and perform exploratory data analysis
- **Model Simulation**: What-if scenario analysis and predictive trace generation
- **Maintenance Planner**: Risk-based scheduling with calendar export capabilities
- **Alerts & Reports**: Automated alerts and PDF report generation
- **Model Explainability**: SHAP analysis and feature importance visualization
- **Chat Assistant**: Natural language interface for maintenance queries
- **Admin Panel**: Model retraining and system configuration

## Project Structure

```
predictive_maintenance/
├── app/
│   └── main.py                 # Main Streamlit application
├── notebooks/
│   ├── data_preprocessing.ipynb # Data preprocessing pipeline
│   ├── vae_lstm_training.ipynb  # VAE-LSTM anomaly detection model
│   └── xgboost_rul_training.ipynb # XGBoost RUL prediction model
├── data/                       # Data storage directory
├── models/                     # Trained model storage
├── utils/                      # Utility functions
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── TODO.md                     # Project progress tracker
```

## Installation

1. **Clone or download the project**:
   ```bash
   cd predictive_maintenance
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app/main.py
   ```

2. **Access the application**:
   - Open your browser and go to `http://localhost:8501`

### Training Models

1. **Data Preprocessing**:
   - Open `notebooks/data_preprocessing.ipynb`
   - Follow the preprocessing steps for your dataset

2. **Train VAE-LSTM Model**:
   - Open `notebooks/vae_lstm_training.ipynb`
   - Run all cells to train the anomaly detection model

3. **Train XGBoost RUL Model**:
   - Open `notebooks/xgboost_rul_training.ipynb`
   - Run all cells to train the RUL prediction model

## Models

### VAE-LSTM Anomaly Detection
- **Architecture**: Variational Autoencoder with LSTM layers
- **Purpose**: Unsupervised anomaly detection in multivariate time series
- **Output**: Reconstruction error scores for anomaly detection

### XGBoost RUL Prediction
- **Algorithm**: Gradient boosting with decision trees
- **Purpose**: Supervised regression for remaining useful life prediction
- **Features**: Engineered time series features and rolling statistics

## Data Format

The application accepts CSV files with the following structure:
- **Time series data**: Timestamped sensor readings
- **Multiple features**: Various sensor measurements
- **Optional labels**: Anomaly labels or RUL values for supervised learning

### Sample Data Structure
```csv
timestamp,sensor1,sensor2,sensor3,temperature,vibration,pressure
2024-01-01 00:00:00,1.2,3.4,5.6,25.0,0.1,101.3
2024-01-01 00:01:00,1.3,3.5,5.7,25.1,0.1,101.2
...
```

## Configuration

### Model Parameters
- **VAE-LSTM**: Configurable hidden dimensions, latent space size, sequence length
- **XGBoost**: Tunable hyperparameters for optimal performance
- **Thresholds**: Adjustable anomaly detection and alert thresholds

### Application Settings
- **Data upload limits**: Configurable file size limits
- **Model update frequency**: Scheduled retraining intervals
- **Alert configurations**: Customizable notification rules

## API Integration

The application supports integration with:
- **OpenAI API**: For chatbot functionality
- **Email services**: For alert notifications
- **Calendar APIs**: For maintenance scheduling
- **Database systems**: For data persistence

## Deployment

### Local Deployment
```bash
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.address", "0.0.0.0"]
```

### Cloud Deployment
- **Streamlit Cloud**: Direct deployment from GitHub
- **Heroku**: Container-based deployment
- **AWS/GCP/Azure**: Scalable cloud deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues:
- Check the documentation
- Open an issue on GitHub
- Contact the development team

## Future Enhancements

- [ ] Real-time data streaming integration
- [ ] Advanced ML models (Transformers, Graph Neural Networks)
- [ ] Multi-asset portfolio optimization
- [ ] Predictive maintenance cost optimization
- [ ] Mobile application companion
- [ ] Integration with SCADA systems
- [ ] Advanced alerting with SMS/email
- [ ] Automated report generation
- [ ] Model performance monitoring
- [ ] A/B testing framework for models
