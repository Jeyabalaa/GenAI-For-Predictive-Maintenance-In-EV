import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np

class PDFReportGenerator:
    def __init__(self):
        """Initialize the PDF report generator."""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Set up custom paragraph styles for the report."""
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )

        self.section_header_style = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=15,
            textColor=colors.darkgreen
        )

        self.normal_style = self.styles['Normal']

    def generate_maintenance_report(self, data: Dict[str, Any], output_path: str = None) -> Optional[bytes]:
        """
        Generate a comprehensive maintenance report PDF.

        Args:
            data: Dictionary containing all report data
            output_path: Path to save the PDF file (optional)

        Returns:
            PDF content as bytes if output_path is None, None otherwise
        """
        if output_path:
            doc = SimpleDocTemplate(output_path, pagesize=A4)
        else:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)

        story = []

        # Title page
        story.extend(self._create_title_page(data))

        # Executive summary
        story.extend(self._create_executive_summary(data))

        # Asset health overview
        story.extend(self._create_asset_health_section(data))

        # Alert analysis
        story.extend(self._create_alerts_section(data))

        # Predictive analytics
        story.extend(self._create_predictive_analytics_section(data))

        # Maintenance recommendations
        story.extend(self._create_recommendations_section(data))

        # Charts and visualizations
        story.extend(self._create_charts_section(data))

        # Build PDF
        doc.build(story)

        if output_path:
            return None
        else:
            buffer.seek(0)
            return buffer.getvalue()

    def _create_title_page(self, data: Dict[str, Any]) -> List:
        """Create the title page of the report."""
        elements = []

        # Title
        title = Paragraph("Predictive Maintenance Report", self.title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.5*inch))

        # Report metadata
        report_date = data.get('report_date', datetime.now().strftime('%Y-%m-%d'))
        generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        metadata_text = f"""
        <b>Report Date:</b> {report_date}<br/>
        <b>Generated:</b> {generated_time}<br/>
        <b>Period:</b> {data.get('period', 'Last 30 days')}<br/>
        <b>Total Assets Monitored:</b> {data.get('total_assets', 'N/A')}<br/>
        """

        elements.append(Paragraph(metadata_text, self.normal_style))
        elements.append(PageBreak())

        return elements

    def _create_executive_summary(self, data: Dict[str, Any]) -> List:
        """Create the executive summary section."""
        elements = []

        elements.append(Paragraph("Executive Summary", self.section_header_style))

        summary_data = data.get('summary', {})

        summary_text = f"""
        This predictive maintenance report provides a comprehensive overview of equipment health,
        maintenance requirements, and predictive analytics for the monitoring period.

        <b>Key Metrics:</b><br/>
        • Total Assets: {summary_data.get('total_assets', 'N/A')}<br/>
        • Healthy Assets: {summary_data.get('healthy_assets', 'N/A')}<br/>
        • Assets Requiring Attention: {summary_data.get('attention_required', 'N/A')}<br/>
        • Critical Alerts: {summary_data.get('critical_alerts', 'N/A')}<br/>
        • Average RUL: {summary_data.get('avg_rul', 'N/A')} days<br/>
        • Maintenance Cost Savings: ${summary_data.get('cost_savings', '0')}<br/>
        """

        elements.append(Paragraph(summary_text, self.normal_style))
        elements.append(Spacer(1, 0.3*inch))

        return elements

    def _create_asset_health_section(self, data: Dict[str, Any]) -> List:
        """Create the asset health overview section."""
        elements = []

        elements.append(Paragraph("Asset Health Overview", self.section_header_style))

        assets_data = data.get('assets', [])
        if assets_data:
            # Create table data
            table_data = [['Asset ID', 'Health Status', 'RUL (days)', 'Risk Score', 'Last Maintenance']]

            for asset in assets_data:
                table_data.append([
                    asset.get('id', 'N/A'),
                    asset.get('health', 'N/A'),
                    str(asset.get('rul', 'N/A')),
                    f"{asset.get('risk_score', 0):.2f}",
                    asset.get('last_maintenance', 'N/A')
                ])

            # Create table
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            elements.append(table)
        else:
            elements.append(Paragraph("No asset data available.", self.normal_style))

        elements.append(Spacer(1, 0.3*inch))
        return elements

    def _create_alerts_section(self, data: Dict[str, Any]) -> List:
        """Create the alerts analysis section."""
        elements = []

        elements.append(Paragraph("Alert Analysis", self.section_header_style))

        alerts_data = data.get('alerts', [])
        if alerts_data:
            # Alert summary
            alert_counts = {}
            for alert in alerts_data:
                severity = alert.get('Severity', 'Unknown')
                alert_counts[severity] = alert_counts.get(severity, 0) + 1

            summary_text = "<b>Alert Summary:</b><br/>"
            for severity, count in alert_counts.items():
                summary_text += f"• {severity}: {count} alerts<br/>"
            summary_text += f"• Total: {len(alerts_data)} alerts"

            elements.append(Paragraph(summary_text, self.normal_style))
            elements.append(Spacer(1, 0.2*inch))

            # Recent alerts table
            table_data = [['Time', 'Asset ID', 'Alert Type', 'Severity', 'Description']]

            # Show last 10 alerts
            for alert in alerts_data[-10:]:
                table_data.append([
                    alert.get('Time', 'N/A'),
                    alert.get('Asset ID', 'N/A'),
                    alert.get('Alert Type', 'N/A'),
                    alert.get('Severity', 'N/A'),
                    alert.get('Description', 'N/A')
                ])

            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            elements.append(table)
        else:
            elements.append(Paragraph("No alerts recorded in this period.", self.normal_style))

        elements.append(Spacer(1, 0.3*inch))
        return elements

    def _create_predictive_analytics_section(self, data: Dict[str, Any]) -> List:
        """Create the predictive analytics section."""
        elements = []

        elements.append(Paragraph("Predictive Analytics", self.section_header_style))

        analytics_data = data.get('analytics', {})

        analytics_text = f"""
        <b>Model Performance:</b><br/>
        • Anomaly Detection Accuracy: {analytics_data.get('anomaly_accuracy', 'N/A')}<br/>
        • RUL Prediction MAE: {analytics_data.get('rul_mae', 'N/A')} days<br/>
        • False Positive Rate: {analytics_data.get('false_positive_rate', 'N/A')}<br/>
        • Model Confidence: {analytics_data.get('model_confidence', 'N/A')}<br/>

        <b>Predictions Summary:</b><br/>
        • Assets predicted to fail within 30 days: {analytics_data.get('assets_fail_30d', 'N/A')}<br/>
        • Assets predicted to fail within 90 days: {analytics_data.get('assets_fail_90d', 'N/A')}<br/>
        • Average predicted RUL: {analytics_data.get('avg_predicted_rul', 'N/A')} days<br/>
        """

        elements.append(Paragraph(analytics_text, self.normal_style))
        elements.append(Spacer(1, 0.3*inch))

        return elements

    def _create_recommendations_section(self, data: Dict[str, Any]) -> List:
        """Create the maintenance recommendations section."""
        elements = []

        elements.append(Paragraph("Maintenance Recommendations", self.section_header_style))

        recommendations = data.get('recommendations', [])

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                rec_text = f"<b>{i}. {rec.get('title', 'Recommendation')}</b><br/>{rec.get('description', '')}<br/>• Priority: {rec.get('priority', 'Medium')}<br/>• Estimated Cost: ${rec.get('cost', 'TBD')}<br/>• Timeline: {rec.get('timeline', 'TBD')}"
                elements.append(Paragraph(rec_text, self.normal_style))
                elements.append(Spacer(1, 0.15*inch))
        else:
            elements.append(Paragraph("No specific recommendations at this time.", self.normal_style))

        elements.append(Spacer(1, 0.3*inch))
        return elements

    def _create_charts_section(self, data: Dict[str, Any]) -> List:
        """Create the charts and visualizations section."""
        elements = []

        elements.append(Paragraph("Charts and Visualizations", self.section_header_style))

        # Placeholder for charts - in a real implementation, you would generate
        # matplotlib/seaborn plots and convert them to images for the PDF
        charts_text = """
        Charts and visualizations would be included here, such as:
        • Asset health distribution pie chart
        • RUL distribution histogram
        • Alert trends over time
        • Feature importance plots
        • Prediction accuracy metrics

        (Implementation would require matplotlib chart generation and image embedding)
        """

        elements.append(Paragraph(charts_text, self.normal_style))
        elements.append(Spacer(1, 0.3*inch))

        return elements

    def generate_quick_report(self, alerts_df: pd.DataFrame, assets_df: pd.DataFrame,
                            predictions: Dict[str, Any], output_path: str = None) -> Optional[bytes]:
        """
        Generate a quick summary report for immediate alerts.

        Args:
            alerts_df: DataFrame with current alerts
            assets_df: DataFrame with asset information
            predictions: Dictionary with prediction results
            output_path: Path to save the PDF file (optional)

        Returns:
            PDF content as bytes if output_path is None, None otherwise
        """
        # Prepare data for the main report generator
        report_data = {
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'period': 'Current Status',
            'total_assets': len(assets_df) if not assets_df.empty else 0,
            'summary': {
                'total_assets': len(assets_df) if not assets_df.empty else 0,
                'critical_alerts': len(alerts_df[alerts_df['Severity'] == 'Critical']) if not alerts_df.empty else 0,
                'avg_rul': predictions.get('avg_rul', 'N/A')
            },
            'alerts': alerts_df.to_dict('records') if not alerts_df.empty else [],
            'assets': assets_df.to_dict('records') if not assets_df.empty else [],
            'analytics': predictions
        }

        return self.generate_maintenance_report(report_data, output_path)
