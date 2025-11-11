import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertNotificationSystem:
    def __init__(self, email_config: Dict[str, str] = None, sms_config: Dict[str, str] = None):
        """
        Initialize the alert notification system.

        Args:
            email_config: Dictionary with email settings (smtp_server, smtp_port, username, password, from_email)
            sms_config: Dictionary with SMS settings (api_key, api_url, sender_id)
        """
        self.email_config = email_config or {}
        self.sms_config = sms_config or {}

        # Default email configuration from environment variables
        if not self.email_config:
            self.email_config = {
                'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                'smtp_port': int(os.getenv('SMTP_PORT', '587')),
                'username': os.getenv('EMAIL_USERNAME'),
                'password': os.getenv('EMAIL_PASSWORD'),
                'from_email': os.getenv('FROM_EMAIL')
            }

        # Default SMS configuration (Twilio example)
        if not self.sms_config:
            self.sms_config = {
                'account_sid': os.getenv('TWILIO_ACCOUNT_SID'),
                'auth_token': os.getenv('TWILIO_AUTH_TOKEN'),
                'from_number': os.getenv('TWILIO_FROM_NUMBER'),
                'api_url': 'https://api.twilio.com/2010-04-01/Accounts/{}/Messages.json'
            }

    def send_email_alert(self, to_emails: List[str], subject: str, message: str,
                        alert_data: Dict[str, Any] = None) -> bool:
        """
        Send email alert notification.

        Args:
            to_emails: List of recipient email addresses
            subject: Email subject
            message: Email message body
            alert_data: Additional alert data for enhanced message

        Returns:
            bool: True if email sent successfully, False otherwise
        """
        if not all(self.email_config.values()):
            logger.error("Email configuration incomplete")
            return False

        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = f"🚨 Predictive Maintenance Alert: {subject}"

            # Enhanced message with alert details
            enhanced_message = self._format_alert_message(message, alert_data)
            msg.attach(MIMEText(enhanced_message, 'html'))

            # Connect to SMTP server
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])

            # Send email
            text = msg.as_string()
            server.sendmail(self.email_config['from_email'], to_emails, text)
            server.quit()

            logger.info(f"Email alert sent successfully to {len(to_emails)} recipients")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def send_sms_alert(self, to_numbers: List[str], message: str,
                      alert_data: Dict[str, Any] = None) -> bool:
        """
        Send SMS alert notification using Twilio.

        Args:
            to_numbers: List of recipient phone numbers
            message: SMS message body
            alert_data: Additional alert data

        Returns:
            bool: True if SMS sent successfully, False otherwise
        """
        if not all([self.sms_config.get('account_sid'), self.sms_config.get('auth_token'),
                   self.sms_config.get('from_number')]):
            logger.error("SMS configuration incomplete")
            return False

        try:
            # Enhanced message for SMS (shorter)
            enhanced_message = self._format_sms_message(message, alert_data)

            success_count = 0
            for number in to_numbers:
                payload = {
                    'From': self.sms_config['from_number'],
                    'To': number,
                    'Body': enhanced_message
                }

                response = requests.post(
                    self.sms_config['api_url'].format(self.sms_config['account_sid']),
                    auth=(self.sms_config['account_sid'], self.sms_config['auth_token']),
                    data=payload
                )

                if response.status_code == 201:
                    success_count += 1
                else:
                    logger.error(f"Failed to send SMS to {number}: {response.text}")

            logger.info(f"SMS alerts sent successfully to {success_count}/{len(to_numbers)} recipients")
            return success_count > 0

        except Exception as e:
            logger.error(f"Failed to send SMS alert: {e}")
            return False

    def send_webhook_notification(self, webhook_url: str, alert_data: Dict[str, Any]) -> bool:
        """
        Send alert notification via webhook.

        Args:
            webhook_url: Webhook URL to send notification to
            alert_data: Alert data to send

        Returns:
            bool: True if webhook sent successfully, False otherwise
        """
        try:
            headers = {'Content-Type': 'application/json'}
            payload = {
                'timestamp': datetime.now().isoformat(),
                'alert_type': 'predictive_maintenance',
                'data': alert_data
            }

            response = requests.post(webhook_url, json=payload, headers=headers, timeout=10)

            if response.status_code in [200, 201, 202]:
                logger.info("Webhook notification sent successfully")
                return True
            else:
                logger.error(f"Webhook failed with status {response.status_code}: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False

    def process_alerts_batch(self, alerts_df: pd.DataFrame, notification_config: Dict[str, Any]) -> Dict[str, int]:
        """
        Process a batch of alerts and send notifications based on configuration.

        Args:
            alerts_df: DataFrame containing alerts
            notification_config: Configuration for notifications (email, sms, webhook settings)

        Returns:
            Dictionary with notification results
        """
        results = {
            'emails_sent': 0,
            'sms_sent': 0,
            'webhooks_sent': 0,
            'errors': 0
        }

        if alerts_df.empty:
            return results

        # Group alerts by severity for batch processing
        severity_groups = alerts_df.groupby('Severity')

        for severity, group in severity_groups:
            if severity.lower() == 'critical':
                # Send immediate notifications for critical alerts
                subject = f"CRITICAL: {len(group)} Maintenance Alerts"
                message = self._create_batch_alert_message(group, severity)

                # Email notifications
                if notification_config.get('email', {}).get('enabled', False):
                    to_emails = notification_config['email'].get('recipients', [])
                    if to_emails:
                        success = self.send_email_alert(to_emails, subject, message, {'alerts': group.to_dict('records')})
                        if success:
                            results['emails_sent'] += 1
                        else:
                            results['errors'] += 1

                # SMS notifications for critical alerts
                if notification_config.get('sms', {}).get('enabled', False):
                    to_numbers = notification_config['sms'].get('recipients', [])
                    if to_numbers:
                        sms_message = f"CRITICAL ALERT: {len(group)} maintenance issues detected. Check email for details."
                        success = self.send_sms_alert(to_numbers, sms_message, {'alert_count': len(group)})
                        if success:
                            results['sms_sent'] += 1
                        else:
                            results['errors'] += 1

                # Webhook notifications
                if notification_config.get('webhook', {}).get('enabled', False):
                    webhook_url = notification_config['webhook'].get('url')
                    if webhook_url:
                        webhook_data = {
                            'severity': severity,
                            'alert_count': len(group),
                            'alerts': group.to_dict('records'),
                            'timestamp': datetime.now().isoformat()
                        }
                        success = self.send_webhook_notification(webhook_url, webhook_data)
                        if success:
                            results['webhooks_sent'] += 1
                        else:
                            results['errors'] += 1

        return results

    def _format_alert_message(self, message: str, alert_data: Dict[str, Any] = None) -> str:
        """Format alert message for email with HTML styling."""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .alert-header {{ background-color: #ff4444; color: white; padding: 10px; border-radius: 5px; }}
                .alert-content {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .alert-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                .alert-table th, .alert-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .alert-table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="alert-header">
                <h2>🔧 Predictive Maintenance Alert</h2>
            </div>
            <div class="alert-content">
                <p><strong>Alert Details:</strong></p>
                <p>{message}</p>
        """

        if alert_data and 'alerts' in alert_data:
            html += """
                <table class="alert-table">
                    <tr>
                        <th>Asset ID</th>
                        <th>Alert Type</th>
                        <th>Severity</th>
                        <th>Value</th>
                        <th>Threshold</th>
                        <th>Time</th>
                    </tr>
            """

            for alert in alert_data['alerts']:
                html += f"""
                    <tr>
                        <td>{alert.get('Asset ID', 'N/A')}</td>
                        <td>{alert.get('Alert Type', 'N/A')}</td>
                        <td>{alert.get('Severity', 'N/A')}</td>
                        <td>{alert.get('Value', 'N/A')}</td>
                        <td>{alert.get('Threshold', 'N/A')}</td>
                        <td>{alert.get('Time', 'N/A')}</td>
                    </tr>
                """

            html += "</table>"

        html += f"""
                <p><em>This alert was generated automatically by the Predictive Maintenance System at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
            </div>
        </body>
        </html>
        """

        return html

    def _format_sms_message(self, message: str, alert_data: Dict[str, Any] = None) -> str:
        """Format concise SMS message."""
        if alert_data and 'alert_count' in alert_data:
            return f"ALERT: {alert_data['alert_count']} maintenance issues detected. Check dashboard for details."
        else:
            return message[:160]  # SMS length limit

    def _create_batch_alert_message(self, alerts_group: pd.DataFrame, severity: str) -> str:
        """Create batch alert message for multiple alerts."""
        asset_count = len(alerts_group['Asset ID'].unique())
        alert_types = alerts_group['Alert Type'].value_counts().to_dict()

        message = f"Multiple {severity.lower()} alerts detected across {asset_count} assets.\n\n"
        message += "Alert Summary:\n"
        for alert_type, count in alert_types.items():
            message += f"- {alert_type}: {count} alerts\n"

        message += f"\nTotal alerts: {len(alerts_group)}"
        message += f"\nPlease check the maintenance dashboard for detailed information."

        return message
