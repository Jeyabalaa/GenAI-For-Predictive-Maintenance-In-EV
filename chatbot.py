import openai
import os
from typing import List, Dict, Any
import json
import pandas as pd
from datetime import datetime

class MaintenanceChatbot:
    def __init__(self, api_key: str = None):
        """Initialize the maintenance chatbot with OpenAI API key."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            # For demo purposes, use a mock implementation if no API key
            self.mock_mode = True
            print("Warning: No OpenAI API key found. Using mock responses for demonstration.")
        else:
            self.mock_mode = False
            openai.api_key = self.api_key

        # System prompt for maintenance context
        self.system_prompt = """
        You are an expert AI assistant specializing in predictive maintenance and industrial equipment monitoring.
        You have deep knowledge of:
        - Mechanical systems, sensors, and industrial equipment
        - Predictive maintenance strategies and best practices
        - Anomaly detection, remaining useful life (RUL) prediction
        - Maintenance scheduling and risk assessment
        - Data analysis and interpretation for maintenance decisions

        When responding to user queries:
        1. Provide actionable, practical advice
        2. Explain technical concepts clearly
        3. Suggest preventive measures and monitoring strategies
        4. Reference industry standards and best practices
        5. Ask clarifying questions when needed
        6. Be proactive in identifying potential issues

        Always maintain a professional, helpful tone and focus on safety and efficiency.
        """

        # Conversation history
        self.conversation_history = []

    def add_context(self, context_type: str, data: Any):
        """Add contextual information about the maintenance system."""
        if context_type == "equipment_data":
            self.equipment_data = data
        elif context_type == "alerts":
            self.active_alerts = data
        elif context_type == "predictions":
            self.predictions = data
        elif context_type == "maintenance_schedule":
            self.maintenance_schedule = data

    def generate_response(self, user_query: str, context: Dict[str, Any] = None) -> str:
        """Generate a response to the user's query using OpenAI or mock responses."""

        if self.mock_mode:
            # Mock responses for demonstration
            mock_responses = {
                "status": "The system is currently monitoring 24 assets with 3 active alerts. Overall health is stable with an average RUL of 156 days.",
                "alerts": "There are currently 3 active alerts: 1 critical temperature alert on Pump A1, 1 warning vibration alert on Motor B2, and 1 low pressure alert on Valve C3.",
                "maintenance": "The next scheduled maintenance is for Pump A1 (bearing inspection) on January 15th, followed by Motor B2 (filter replacement) on January 20th.",
                "recommendations": "Based on current data, I recommend: 1) Immediate inspection of Pump A1 due to critical temperature readings, 2) Schedule vibration analysis for Motor B2, 3) Calibrate pressure sensors on Valve C3."
            }

            # Simple keyword matching for mock responses
            user_lower = user_query.lower()
            if "status" in user_lower or "system" in user_lower:
                response = mock_responses["status"]
            elif "alert" in user_lower:
                response = mock_responses["alerts"]
            elif "maintenance" in user_lower or "schedule" in user_lower:
                response = mock_responses["maintenance"]
            elif "recommend" in user_lower:
                response = mock_responses["recommendations"]
            else:
                response = f"Thank you for your question about '{user_query}'. As an AI maintenance assistant, I'm here to help with predictive maintenance queries. Please ask about system status, alerts, maintenance schedules, or recommendations."

            # Update conversation history
            self.conversation_history.extend([
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": response}
            ])

            return response

        # Build context-aware prompt
        context_info = ""
        if context:
            if 'equipment_data' in context and context['equipment_data'] is not None:
                context_info += f"\nCurrent Equipment Data Summary:\n{self._summarize_equipment_data(context['equipment_data'])}"

            if 'alerts' in context and context['alerts'] is not None:
                context_info += f"\nActive Alerts:\n{self._summarize_alerts(context['alerts'])}"

            if 'predictions' in context and context['predictions'] is not None:
                context_info += f"\nCurrent Predictions:\n{self._summarize_predictions(context['predictions'])}"

        # Create messages for OpenAI API
        messages = [
            {"role": "system", "content": self.system_prompt + context_info},
            {"role": "user", "content": user_query}
        ]

        # Add conversation history (last 5 exchanges)
        if len(self.conversation_history) > 0:
            recent_history = self.conversation_history[-10:]  # Last 5 exchanges (10 messages)
            messages = [{"role": "system", "content": self.system_prompt + context_info}] + recent_history + [{"role": "user", "content": user_query}]

        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                top_p=0.9
            )

            ai_response = response.choices[0].message.content.strip()

            # Update conversation history
            self.conversation_history.extend([
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": ai_response}
            ])

            # Keep only last 10 messages to avoid token limits
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            return ai_response

        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try again or contact support."

    def _summarize_equipment_data(self, data: pd.DataFrame) -> str:
        """Summarize equipment data for context."""
        if data is None or data.empty:
            return "No equipment data available."

        summary = f"Dataset contains {len(data)} records with {len(data.columns)} features.\n"
        summary += f"Time range: {data.index.min()} to {data.index.max()}\n"

        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary += f"Key metrics (latest values):\n"
            latest = data.iloc[-1] if hasattr(data, 'iloc') else data.tail(1).iloc[0]
            for col in numeric_cols[:5]:  # Show first 5 numeric columns
                summary += f"- {col}: {latest[col]:.2f}\n"

        return summary

    def _summarize_alerts(self, alerts: pd.DataFrame) -> str:
        """Summarize active alerts for context."""
        if alerts is None or alerts.empty:
            return "No active alerts."

        summary = f"{len(alerts)} active alerts:\n"
        for _, alert in alerts.iterrows():
            summary += f"- {alert.get('Asset ID', 'Unknown')}: {alert.get('Alert Type', 'Unknown')} ({alert.get('Severity', 'Unknown')} severity)\n"

        return summary

    def _summarize_predictions(self, predictions: Dict[str, Any]) -> str:
        """Summarize predictions for context."""
        if not predictions:
            return "No predictions available."

        summary = "Current predictions:\n"
        if 'rul' in predictions:
            summary += f"- Average RUL: {predictions['rul']:.1f} days\n"
        if 'anomaly_rate' in predictions:
            summary += f"- Current anomaly rate: {predictions['anomaly_rate']:.1f}%\n"
        if 'risk_score' in predictions:
            summary += f"- Overall risk score: {predictions['risk_score']:.2f}\n"

        return summary

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def get_suggestions(self, context: Dict[str, Any] = None) -> List[str]:
        """Generate proactive maintenance suggestions based on current context."""
        suggestions = []

        if context:
            if 'alerts' in context and context['alerts'] is not None and not context['alerts'].empty:
                suggestions.append("Review and address active alerts immediately")
                critical_alerts = context['alerts'][context['alerts']['Severity'] == 'Critical']
                if not critical_alerts.empty:
                    suggestions.append("Critical alerts detected - consider immediate maintenance action")

            if 'predictions' in context and context['predictions'] is not None:
                if context['predictions'].get('anomaly_rate', 0) > 10:
                    suggestions.append("High anomaly rate detected - investigate root causes")
                if context['predictions'].get('rul', 100) < 30:
                    suggestions.append("Equipment approaching end of life - plan replacement")

        # General suggestions
        suggestions.extend([
            "Schedule regular preventive maintenance",
            "Monitor sensor calibration and accuracy",
            "Review historical maintenance data for patterns",
            "Consider implementing condition-based monitoring"
        ])

        return suggestions[:5]  # Return top 5 suggestions

    def generate_summary(self, chat_history: List[Dict[str, str]]) -> str:
        """Generate a summary of the chat conversation."""
        if not chat_history:
            return "No conversation history to summarize."

        # Simple summary for mock mode
        if self.mock_mode:
            return "Chat Summary: The conversation covered system status inquiries, active alerts review, maintenance scheduling questions, and general recommendations for predictive maintenance best practices."

        # For real API mode, could implement more sophisticated summarization
        return "Chat Summary: The conversation covered various aspects of predictive maintenance including system monitoring, alert management, and maintenance planning."
