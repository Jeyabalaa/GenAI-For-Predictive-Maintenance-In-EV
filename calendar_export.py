import pandas as pd
from icalendar import Calendar, Event, vCalAddress, vText
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Any, Optional
import uuid
import os

class CalendarExporter:
    def __init__(self, timezone: str = 'UTC'):
        """
        Initialize the calendar exporter.

        Args:
            timezone: Default timezone for events
        """
        self.timezone = pytz.timezone(timezone)

    def export_maintenance_schedule(self, schedule_df: pd.DataFrame,
                                  output_path: str = None,
                                  calendar_name: str = "Predictive Maintenance Schedule") -> Optional[str]:
        """
        Export maintenance schedule to iCalendar (.ics) format.

        Args:
            schedule_df: DataFrame containing maintenance schedule
            output_path: Path to save the .ics file (optional)
            calendar_name: Name for the calendar

        Returns:
            iCalendar content as string if output_path is None, None otherwise
        """
        # Create calendar
        cal = Calendar()
        cal.add('prodid', '-//Predictive Maintenance System//')
        cal.add('version', '2.0')
        cal.add('name', calendar_name)
        cal.add('description', 'Automated maintenance schedule from predictive maintenance system')

        if schedule_df.empty:
            return None

        # Process each maintenance task
        for _, task in schedule_df.iterrows():
            event = self._create_maintenance_event(task)
            if event:
                cal.add_component(event)

        # Generate iCalendar content
        ics_content = cal.to_ical().decode('utf-8')

        # Save to file if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(ics_content)
            return None
        else:
            return ics_content

    def _create_maintenance_event(self, task: pd.Series) -> Optional[Event]:
        """Create an iCalendar event for a maintenance task."""
        try:
            event = Event()

            # Basic event properties
            event.add('uid', str(uuid.uuid4()))
            event.add('summary', f"Maintenance: {task.get('Task', 'Unknown Task')}")

            # Description
            description = self._format_event_description(task)
            event.add('description', description)

            # Dates - handle different date formats
            due_date = self._parse_date(task.get('Due Date', ''))
            if due_date:
                # Set event to start at 9 AM on due date and last 2 hours
                start_time = due_date.replace(hour=9, minute=0, second=0, microsecond=0)
                end_time = start_time + timedelta(hours=2)

                event.add('dtstart', start_time)
                event.add('dtend', end_time)
            else:
                # If no due date, skip this event
                return None

            # Priority based on task priority
            priority = self._get_priority_value(task.get('Priority', 'Medium'))
            event.add('priority', priority)

            # Location (asset identifier)
            asset_id = task.get('Asset', 'Unknown Asset')
            event.add('location', f"Asset: {asset_id}")

            # Categories
            event.add('categories', ['Maintenance', 'Predictive'])

            # Status
            event.add('status', 'CONFIRMED')

            # Reminder (1 day before)
            alarm = self._create_alarm(start_time)
            if alarm:
                event.add_component(alarm)

            return event

        except Exception as e:
            print(f"Error creating event for task {task.get('Task', 'Unknown')}: {e}")
            return None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string into datetime object."""
        if not date_str or date_str == 'N/A':
            return None

        try:
            # Try different date formats
            formats = [
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%d-%m-%Y',
                '%Y/%m/%d'
            ]

            for fmt in formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    # Localize to timezone
                    return self.timezone.localize(parsed_date)
                except ValueError:
                    continue

            return None

        except Exception:
            return None

    def _format_event_description(self, task: pd.Series) -> str:
        """Format detailed event description."""
        description_parts = []

        # Task details
        description_parts.append(f"Task: {task.get('Task', 'N/A')}")
        description_parts.append(f"Asset: {task.get('Asset', 'N/A')}")
        description_parts.append(f"Priority: {task.get('Priority', 'N/A')}")

        # Cost information
        cost = task.get('Estimated Cost', 'N/A')
        if cost != 'N/A':
            description_parts.append(f"Estimated Cost: {cost}")

        # Additional notes
        if 'Notes' in task and task['Notes']:
            description_parts.append(f"Notes: {task['Notes']}")

        # Contact information placeholder
        description_parts.append("Contact: Maintenance Team")
        description_parts.append("System: Predictive Maintenance Dashboard")

        return '\n'.join(description_parts)

    def _get_priority_value(self, priority: str) -> int:
        """Convert priority string to iCalendar priority value (1-9)."""
        priority_map = {
            'Critical': 1,
            'High': 2,
            'Medium': 5,
            'Low': 7,
            'Routine': 9
        }
        return priority_map.get(priority.lower(), 5)

    def _create_alarm(self, event_start: datetime) -> Optional[Any]:
        """Create a reminder alarm for the event."""
        try:
            from icalendar import Alarm

            alarm = Alarm()
            alarm.add('action', 'DISPLAY')
            alarm.add('description', 'Maintenance Task Reminder')
            alarm.add('trigger', timedelta(days=-1))  # 1 day before

            return alarm

        except Exception:
            return None

    def export_to_csv(self, schedule_df: pd.DataFrame, output_path: str = None) -> Optional[str]:
        """
        Export maintenance schedule to CSV format.

        Args:
            schedule_df: DataFrame containing maintenance schedule
            output_path: Path to save the CSV file (optional)

        Returns:
            CSV content as string if output_path is None, None otherwise
        """
        if schedule_df.empty:
            return None

        # Prepare CSV data
        csv_df = schedule_df.copy()

        # Add additional columns for calendar integration
        csv_df['Event Title'] = csv_df.apply(
            lambda row: f"Maintenance: {row.get('Task', 'Unknown Task')}", axis=1
        )

        csv_df['Start Date'] = csv_df['Due Date']  # Assuming Due Date is the start date
        csv_df['Start Time'] = '09:00:00'  # Default start time
        csv_df['End Time'] = '11:00:00'    # Default end time (2 hours later)
        csv_df['All Day Event'] = False
        csv_df['Description'] = csv_df.apply(self._format_csv_description, axis=1)
        csv_df['Location'] = csv_df.apply(
            lambda row: f"Asset: {row.get('Asset', 'Unknown Asset')}", axis=1
        )
        csv_df['Private'] = False

        # Reorder columns for better CSV structure
        column_order = [
            'Event Title', 'Start Date', 'Start Time', 'End Time', 'All Day Event',
            'Description', 'Location', 'Private', 'Asset', 'Task', 'Priority',
            'Estimated Cost', 'Due Date'
        ]

        # Keep only existing columns
        available_columns = [col for col in column_order if col in csv_df.columns]
        csv_df = csv_df[available_columns]

        # Generate CSV content
        csv_content = csv_df.to_csv(index=False)

        # Save to file if path provided
        if output_path:
            csv_df.to_csv(output_path, index=False)
            return None
        else:
            return csv_content

    def _format_csv_description(self, task: pd.Series) -> str:
        """Format description for CSV export."""
        description_parts = [
            f"Task: {task.get('Task', 'N/A')}",
            f"Asset: {task.get('Asset', 'N/A')}",
            f"Priority: {task.get('Priority', 'N/A')}",
            f"Estimated Cost: {task.get('Estimated Cost', 'N/A')}",
            "Generated by Predictive Maintenance System"
        ]

        return ' | '.join(description_parts)

    def create_recurring_maintenance_event(self, asset_id: str, task_name: str,
                                         frequency_days: int, start_date: datetime,
                                         duration_hours: int = 2) -> Event:
        """
        Create a recurring maintenance event.

        Args:
            asset_id: Asset identifier
            task_name: Name of the maintenance task
            frequency_days: How often the task should repeat (in days)
            start_date: When the recurring schedule starts
            duration_hours: How long each maintenance session takes

        Returns:
            iCalendar Event with recurrence rules
        """
        event = Event()

        # Basic properties
        event.add('uid', str(uuid.uuid4()))
        event.add('summary', f"Recurring Maintenance: {task_name}")
        event.add('description', f"Regular maintenance for asset {asset_id}")
        event.add('location', f"Asset: {asset_id}")

        # Timing
        start_time = start_date.replace(hour=9, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(hours=duration_hours)

        event.add('dtstart', start_time)
        event.add('dtend', end_time)

        # Recurrence rule (repeat every frequency_days)
        rrule = f"FREQ=DAILY;INTERVAL={frequency_days}"
        event.add('rrule', rrule)

        # Priority and categories
        event.add('priority', 5)  # Medium priority
        event.add('categories', ['Maintenance', 'Recurring', 'Predictive'])
        event.add('status', 'CONFIRMED')

        return event
