import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import utility modules
from utils.chatbot import MaintenanceChatbot
from utils.shap_explainability import ModelExplainer
from utils.alerts_notifications import AlertNotificationSystem
from utils.pdf_reports import PDFReportGenerator
from utils.calendar_export import CalendarExporter

# Set page configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'anomaly_threshold' not in st.session_state:
    st.session_state.anomaly_threshold = 0.8

def main():
    # Sidebar navigation
    st.sidebar.markdown('<div class="sidebar-header">🔧 Predictive Maintenance</div>', unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Dashboard", "📊 Data Explorer", "🎯 Model Simulation", "📅 Maintenance Planner",
         "🚨 Alerts & Reports", "🔍 Model Explainability", "💬 Chat Assistant", "⚙️ Admin"]
    )

    # Main content
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📊 Data Explorer":
        show_data_explorer()
    elif page == "🎯 Model Simulation":
        show_model_simulation()
    elif page == "📅 Maintenance Planner":
        show_maintenance_planner()
    elif page == "🚨 Alerts & Reports":
        show_alerts_reports()
    elif page == "🔍 Model Explainability":
        show_model_explainability()
    elif page == "💬 Chat Assistant":
        show_chat_assistant()
    elif page == "⚙️ Admin":
        show_admin()

def show_dashboard():
    st.markdown('<div class="main-header">Predictive Maintenance Dashboard</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Assets", "24", "↗️ 2")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Active Alerts", "3", "⚠️")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg RUL", "156 days", "↗️ 12")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Anomaly Rate", "4.2%", "↘️ 0.8%")
        st.markdown('</div>', unsafe_allow_html=True)

    # Charts section
    st.subheader("System Overview")

    col1, col2 = st.columns(2)

    with col1:
        # Asset health distribution
        fig = px.pie(
            values=[65, 25, 10],
            names=['Healthy', 'Warning', 'Critical'],
            title='Asset Health Distribution',
            color_discrete_sequence=['#00ff00', '#ffff00', '#ff0000']
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # RUL distribution
        np.random.seed(42)
        rul_data = np.random.normal(150, 50, 100)
        fig = px.histogram(
            rul_data,
            title='Remaining Useful Life Distribution',
            labels={'value': 'RUL (days)'},
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig, use_container_width=True)

    # Recent alerts
    st.subheader("Recent Alerts")
    alerts_data = pd.DataFrame({
        'Asset': ['Pump A1', 'Motor B2', 'Valve C3'],
        'Alert Type': ['High Vibration', 'Temperature Spike', 'Pressure Drop'],
        'Severity': ['High', 'Medium', 'Low'],
        'Time': ['2 hours ago', '4 hours ago', '6 hours ago']
    })
    st.dataframe(alerts_data, use_container_width=True)

def show_data_explorer():
    st.markdown('<div class="main-header">Data Explorer</div>', unsafe_allow_html=True)

    # File upload section
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV or ZIP file", type=['csv', 'zip'])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.data = pd.read_csv(uploaded_file)
                st.success(f"Loaded {uploaded_file.name} successfully!")
            elif uploaded_file.name.endswith('.zip'):
                # Handle ZIP files (would need implementation)
                st.info("ZIP file handling would be implemented here")
        except Exception as e:
            st.error(f"Error loading file: {e}")

    # Data preview and statistics
    if st.session_state.data is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head(), use_container_width=True)

        st.subheader("Data Statistics")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Shape:**", st.session_state.data.shape)
            st.write("**Columns:**", list(st.session_state.data.columns))

        with col2:
            numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
            st.write("**Numeric Columns:**", len(numeric_cols))
            st.write("**Missing Values:**", st.session_state.data.isnull().sum().sum())

        # Basic EDA plots
        if len(numeric_cols) > 0:
            st.subheader("Exploratory Data Analysis")

            # Correlation heatmap
            if len(numeric_cols) > 1:
                corr = st.session_state.data[numeric_cols].corr()
                fig = px.imshow(corr, title="Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)

            # Distribution plots
            selected_col = st.selectbox("Select column for distribution plot", numeric_cols)
            fig = px.histogram(st.session_state.data, x=selected_col, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)

def show_model_simulation():
    st.markdown('<div class="main-header">Model Simulation</div>', unsafe_allow_html=True)

    st.info("Model simulation features would be implemented here, including:")
    st.write("- What-if scenario analysis")
    st.write("- Parameter sensitivity testing")
    st.write("- Predictive trace generation")
    st.write("- Risk probability visualization")

    # Placeholder for simulation controls
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Simulation Parameters")
        scenario = st.selectbox("Select Scenario", ["Normal Operation", "High Load", "Degradation", "Failure Mode"])
        duration = st.slider("Simulation Duration (hours)", 1, 168, 24)
        confidence = st.slider("Confidence Level", 0.5, 0.99, 0.95)

    with col2:
        st.subheader("Results")
        if st.button("Run Simulation"):
            st.success("Simulation completed!")
            # Placeholder results
            st.metric("Predicted RUL", "142 hours")
            st.metric("Failure Probability", "12.3%")

def show_maintenance_planner():
    st.markdown('<div class="main-header">Maintenance Planner</div>', unsafe_allow_html=True)

    st.info("Maintenance planning features include:")
    st.write("- Risk-based maintenance scheduling")
    st.write("- Resource optimization")
    st.write("- Calendar integration")
    st.write("- Cost-benefit analysis")

    # Sample maintenance schedule
    st.subheader("Upcoming Maintenance")
    schedule_data = pd.DataFrame({
        'Asset': ['Pump A1', 'Motor B2', 'Valve C3', 'Compressor D1'],
        'Task': ['Routine Inspection', 'Bearing Replacement', 'Calibration', 'Filter Change'],
        'Priority': ['High', 'Medium', 'Low', 'Medium'],
        'Due Date': ['2024-01-15', '2024-01-20', '2024-01-25', '2024-02-01'],
        'Estimated Cost': ['$500', '$2,500', '$300', '$800']
    })
    st.dataframe(schedule_data, use_container_width=True)

    # Calendar export functionality
    st.subheader("Export to Calendar")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📅 Export to iCalendar (.ics)"):
            try:
                calendar_exporter = CalendarExporter()
                ics_content = calendar_exporter.export_maintenance_schedule(schedule_data)

                if ics_content:
                    st.download_button(
                        label="📥 Download Calendar File",
                        data=ics_content,
                        file_name=f"maintenance_schedule_{datetime.now().strftime('%Y%m%d')}.ics",
                        mime="text/calendar"
                    )
                    st.success("iCalendar file generated successfully!")
                else:
                    st.error("Failed to generate calendar file.")
            except Exception as e:
                st.error(f"Error generating calendar file: {e}")

    with col2:
        if st.button("📊 Export to CSV"):
            try:
                calendar_exporter = CalendarExporter()
                csv_content = calendar_exporter.export_to_csv(schedule_data)

                if csv_content:
                    st.download_button(
                        label="📥 Download CSV File",
                        data=csv_content,
                        file_name=f"maintenance_schedule_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    st.success("CSV file generated successfully!")
                else:
                    st.error("Failed to generate CSV file.")
            except Exception as e:
                st.error(f"Error generating CSV file: {e}")

    # Maintenance recommendations
    st.subheader("AI-Generated Recommendations")
    if st.button("🔍 Generate Maintenance Recommendations"):
        try:
            # Initialize chatbot for recommendations
            chatbot = MaintenanceChatbot()

            # Sample context
            context = {
                'alerts': pd.DataFrame({
                    'Asset ID': ['A001', 'B002'],
                    'Alert Type': ['Vibration', 'Temperature'],
                    'Severity': ['Critical', 'Warning']
                }),
                'predictions': {
                    'avg_rul': 45,
                    'anomaly_rate': 15.2
                }
            }

            recommendations = chatbot.get_suggestions(context)

            st.subheader("Recommended Actions:")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")

        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            st.info("Make sure to set your OpenAI API key in the environment variables.")

def show_alerts_reports():
    st.markdown('<div class="main-header">Alerts & Reports</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Active Alerts", "Reports", "Notifications"])

    with tab1:
        st.subheader("Active Alerts")
        alerts = pd.DataFrame({
            'Asset ID': ['A001', 'B002', 'C003'],
            'Alert Type': ['Temperature', 'Vibration', 'Pressure'],
            'Severity': ['Critical', 'Warning', 'Warning'],
            'Value': ['95°C', '12.5 mm/s', '2.1 bar'],
            'Threshold': ['90°C', '10 mm/s', '2.5 bar'],
            'Time': ['10:30 AM', '09:15 AM', '08:45 AM']
        })
        st.dataframe(alerts, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate PDF Report"):
                try:
                    # Generate PDF report
                    pdf_generator = PDFReportGenerator()
                    report_data = {
                        'report_date': datetime.now().strftime('%Y-%m-%d'),
                        'period': 'Current Status',
                        'total_assets': 24,
                        'summary': {
                            'total_assets': 24,
                            'critical_alerts': len(alerts[alerts['Severity'] == 'Critical']),
                            'avg_rul': 156
                        },
                        'alerts': alerts.to_dict('records'),
                        'assets': [],
                        'analytics': {'avg_rul': 156}
                    }

                    pdf_content = pdf_generator.generate_maintenance_report(report_data)
                    if pdf_content:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_content,
                            file_name=f"maintenance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    st.success("PDF report generated successfully!")
                except Exception as e:
                    st.error(f"Error generating PDF report: {e}")

        with col2:
            if st.button("Send Alert Notifications"):
                try:
                    # Initialize notification system
                    alert_system = AlertNotificationSystem()

                    # Send notifications for critical alerts
                    critical_alerts = alerts[alerts['Severity'] == 'Critical']
                    if not critical_alerts.empty:
                        notification_config = {
                            'email': {
                                'enabled': True,
                                'recipients': ['maintenance@example.com']  # Configure as needed
                            },
                            'sms': {
                                'enabled': True,
                                'recipients': ['+1234567890']  # Configure as needed
                            }
                        }

                        results = alert_system.process_alerts_batch(critical_alerts, notification_config)
                        st.success(f"Notifications sent: {results['emails_sent']} emails, {results['sms_sent']} SMS")
                    else:
                        st.info("No critical alerts to notify about.")
                except Exception as e:
                    st.error(f"Error sending notifications: {e}")

    with tab2:
        st.subheader("Report Generation")
        report_type = st.selectbox("Report Type", ["Daily Summary", "Weekly Analysis", "Monthly Report", "Custom"])
        date_range = st.date_input("Date Range", [datetime.now() - timedelta(days=7), datetime.now()])

        if st.button("Generate Report"):
            try:
                # Generate comprehensive report
                pdf_generator = PDFReportGenerator()
                report_data = {
                    'report_date': datetime.now().strftime('%Y-%m-%d'),
                    'period': f"{date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}",
                    'total_assets': 24,
                    'summary': {
                        'total_assets': 24,
                        'healthy_assets': 16,
                        'attention_required': 5,
                        'critical_alerts': 3,
                        'avg_rul': 156,
                        'cost_savings': 15000
                    },
                    'alerts': [],
                    'assets': [
                        {'id': 'A001', 'health': 'Critical', 'rul': 15, 'risk_score': 0.95, 'last_maintenance': '2024-01-01'},
                        {'id': 'B002', 'health': 'Warning', 'rul': 45, 'risk_score': 0.75, 'last_maintenance': '2024-01-10'},
                        {'id': 'C003', 'health': 'Healthy', 'rul': 180, 'risk_score': 0.25, 'last_maintenance': '2024-01-15'}
                    ],
                    'analytics': {
                        'anomaly_accuracy': '94.2%',
                        'rul_mae': 12.5,
                        'false_positive_rate': '3.1%',
                        'model_confidence': '87.3%',
                        'assets_fail_30d': 2,
                        'assets_fail_90d': 5,
                        'avg_predicted_rul': 142
                    },
                    'recommendations': [
                        {'title': 'Replace Pump A1 Bearings', 'description': 'Critical vibration levels detected', 'priority': 'High', 'cost': '2500', 'timeline': 'Within 1 week'},
                        {'title': 'Calibrate Temperature Sensors', 'description': 'Drift detected in multiple sensors', 'priority': 'Medium', 'cost': '800', 'timeline': 'Within 2 weeks'},
                        {'title': 'Schedule Routine Inspection', 'description': 'Quarterly maintenance due', 'priority': 'Low', 'cost': '300', 'timeline': 'Within 1 month'}
                    ]
                }

                pdf_content = pdf_generator.generate_maintenance_report(report_data)
                if pdf_content:
                    st.download_button(
                        label="📥 Download Report",
                        data=pdf_content,
                        file_name=f"{report_type.lower().replace(' ', '_')}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                st.success("Report generated successfully!")
            except Exception as e:
                st.error(f"Error generating report: {e}")

    with tab3:
        st.subheader("Notification Settings")

        st.write("Configure automated notifications for alerts:")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Email Notifications")
            email_enabled = st.checkbox("Enable Email Alerts", value=False)
            if email_enabled:
                email_recipients = st.text_area("Email Recipients (one per line)", "maintenance@example.com\nmanager@example.com")
                smtp_server = st.text_input("SMTP Server", "smtp.gmail.com")
                smtp_port = st.number_input("SMTP Port", value=587)
                email_username = st.text_input("Email Username")
                email_password = st.text_input("Email Password", type="password")

        with col2:
            st.subheader("SMS Notifications")
            sms_enabled = st.checkbox("Enable SMS Alerts", value=False)
            if sms_enabled:
                sms_recipients = st.text_area("SMS Recipients (one per line)", "+1234567890\n+0987654321")
                twilio_sid = st.text_input("Twilio Account SID")
                twilio_token = st.text_input("Twilio Auth Token", type="password")
                twilio_from = st.text_input("Twilio From Number")

        if st.button("Save Notification Settings"):
            st.success("Notification settings saved successfully!")
            # In a real implementation, these would be saved to a configuration file or database

def show_model_explainability():
    st.markdown('<div class="main-header">Model Explainability</div>', unsafe_allow_html=True)

    st.info("Model explainability features include:")
    st.write("- Feature importance analysis")
    st.write("- SHAP value visualizations")
    st.write("- Partial dependence plots")
    st.write("- Model performance metrics")

    # Initialize explainer
    explainer = ModelExplainer()

    tab1, tab2, tab3 = st.tabs(["Feature Importance", "SHAP Analysis", "Model Performance"])

    with tab1:
        st.subheader("Feature Importance Analysis")

        # Sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        features = ['Temperature', 'Vibration', 'Pressure', 'Current', 'Voltage', 'Flow_Rate', 'Power']
        X_train = pd.DataFrame(np.random.randn(n_samples, len(features)), columns=features)
        X_test = pd.DataFrame(np.random.randn(100, len(features)), columns=features)

        # Simulate XGBoost model (placeholder)
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, np.random.randn(n_samples))

            # Explain model
            explanations = explainer.explain_xgboost_model(model, X_train, X_test)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Feature Importance")
                feature_imp = explanations['feature_importance']
                fig = px.bar(
                    x=list(feature_imp.keys()),
                    y=list(feature_imp.values()),
                    title="XGBoost Feature Importance",
                    labels={'x': 'Features', 'y': 'Importance'}
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("SHAP Summary Plot")
                if 'summary_plot' in explanations:
                    st.plotly_chart(explanations['summary_plot'], use_container_width=True)
                else:
                    st.write("SHAP summary plot not available")

        except ImportError:
            st.warning("XGBoost not installed. Install with: pip install xgboost")
            # Fallback to sample data
            features = ['Temperature', 'Vibration', 'Pressure', 'Current', 'Voltage']
            importance = [0.35, 0.28, 0.18, 0.12, 0.07]
            fig = px.bar(x=features, y=importance, title="Sample Feature Importance")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("SHAP Value Analysis")

        st.write("SHAP (SHapley Additive exPlanations) values show how each feature contributes to individual predictions.")

        # Sample SHAP analysis
        try:
            if 'explanations' in locals() and 'shap_values' in explanations:
                st.subheader("Waterfall Plot (Sample Prediction)")
                if 'waterfall_plot' in explanations:
                    st.plotly_chart(explanations['waterfall_plot'], use_container_width=True)

                st.subheader("SHAP Value Distribution")
                shap_vals = explanations['shap_values']
                if shap_vals is not None:
                    fig = px.histogram(
                        x=shap_vals.flatten() if hasattr(shap_vals, 'flatten') else shap_vals.tolist(),
                        title="SHAP Value Distribution",
                        labels={'x': 'SHAP Values'},
                        color_discrete_sequence=['lightcoral']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("SHAP values not available for distribution plot")
            else:
                st.write("Load a trained model to see SHAP analysis")
        except Exception as e:
            st.error(f"Error in SHAP analysis: {e}")

    with tab3:
        st.subheader("Model Performance Metrics")

        # Sample performance metrics
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MAE', 'RMSE'],
            'Anomaly Detection': [94.2, 89.5, 91.8, 90.6, None, None],
            'RUL Prediction': [None, None, None, None, 12.5, 18.7]
        }

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)

        # Performance visualization
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model Accuracy Over Time")
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            accuracy = 85 + 10 * np.sin(np.linspace(0, 4*np.pi, 30)) + np.random.normal(0, 2, 30)
            fig = px.line(x=dates, y=accuracy, title="Model Accuracy Trend")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Prediction Error Distribution")
            errors = np.random.normal(0, 15, 1000)
            fig = px.histogram(errors, title="Prediction Error Distribution", labels={'value': 'Error'})
            st.plotly_chart(fig, use_container_width=True)

def show_chat_assistant():
    st.markdown('<div class="main-header">AI Chat Assistant</div>', unsafe_allow_html=True)

    st.info("AI-powered chat assistant for:")
    st.write("- Natural language queries about maintenance data")
    st.write("- Predictive simulations based on text input")
    st.write("- Maintenance recommendations")
    st.write("- Troubleshooting guidance")

    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize chatbot
    chatbot = MaintenanceChatbot()

    # Display chat history
    st.subheader("Chat History")
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"🤖 **Assistant:** {message['content']}")
            st.markdown("---")

    # Chat input
    with st.form(key='chat_form'):
        user_input = st.text_area(
            "Ask me anything about your predictive maintenance system...",
            height=100,
            key="chat_input"
        )

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            submit_button = st.form_submit_button("Send Message")

        with col2:
            if st.form_submit_button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

        with col3:
            if st.form_submit_button("Generate Report Summary"):
                try:
                    # Generate a summary report based on chat context
                    if st.session_state.chat_history:
                        summary = chatbot.generate_summary(st.session_state.chat_history)
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': f"**Summary Report:**\n{summary}"
                        })
                        st.rerun()
                    else:
                        st.warning("No chat history to summarize.")
                except Exception as e:
                    st.error(f"Error generating summary: {e}")

    if submit_button and user_input.strip():
        try:
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })

            # Get chatbot response
            response = chatbot.generate_response(user_input, context=st.session_state.chat_history)

            # Add assistant response to history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })

            st.rerun()

        except Exception as e:
            st.error(f"Error communicating with chatbot: {e}")
            st.info("Make sure to set your OpenAI API key in the environment variables.")

    # Quick action buttons
    st.subheader("Quick Actions")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📊 System Status"):
            quick_query = "What is the current status of the maintenance system?"
            try:
                response = chatbot.generate_response(quick_query)
                st.session_state.chat_history.append({'role': 'user', 'content': quick_query})
                st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        if st.button("🚨 Active Alerts"):
            quick_query = "Show me all active alerts and their severity levels."
            try:
                response = chatbot.generate_response(quick_query)
                st.session_state.chat_history.append({'role': 'user', 'content': quick_query})
                st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    with col3:
        if st.button("📅 Maintenance Schedule"):
            quick_query = "What maintenance activities are scheduled for the next week?"
            try:
                response = chatbot.generate_response(quick_query)
                st.session_state.chat_history.append({'role': 'user', 'content': quick_query})
                st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    with col4:
        if st.button("🔧 Recommendations"):
            quick_query = "What are the top maintenance recommendations right now?"
            try:
                response = chatbot.generate_response(quick_query)
                st.session_state.chat_history.append({'role': 'user', 'content': quick_query})
                st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

def show_admin():
    st.markdown('<div class="main-header">Admin Panel</div>', unsafe_allow_html=True)

    st.info("Admin features would include:")
    st.write("- Model retraining and versioning")
    st.write("- System configuration")
    st.write("- User management")
    st.write("- Performance monitoring")

    # Model management
    st.subheader("Model Management")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Retrain Models"):
            st.success("Model retraining initiated!")

    with col2:
        if st.button("Update Thresholds"):
            st.success("Thresholds updated!")

if __name__ == "__main__":
    main()
