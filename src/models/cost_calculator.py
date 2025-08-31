"""
Cost Calculator

Calculates costs for model usage based on token consumption and pricing tiers.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

from .model_registry import ModelRegistry, ModelConfig


class BillingTier(str, Enum):
    """Billing tiers with different pricing structures."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass
class UsageRecord:
    """Record of model usage for cost tracking."""
    timestamp: datetime
    model_id: str
    input_tokens: int
    output_tokens: int
    cost: float
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostSummary:
    """Summary of costs over a period."""
    total_cost: float
    total_input_tokens: int
    total_output_tokens: int
    total_requests: int
    cost_by_model: Dict[str, float]
    period_start: datetime
    period_end: datetime
    average_cost_per_request: float
    most_expensive_model: Optional[str] = None
    cheapest_model: Optional[str] = None


class CostCalculator:
    """Calculates and tracks costs for model usage."""
    
    def __init__(self, 
                 billing_tier: BillingTier = BillingTier.BASIC,
                 track_usage: bool = True,
                 usage_file: Optional[str] = None):
        """
        Initialize the cost calculator.
        
        Args:
            billing_tier: Billing tier for pricing adjustments
            track_usage: Whether to track usage history
            usage_file: File to persist usage records (optional)
        """
        self.billing_tier = billing_tier
        self.track_usage = track_usage
        self.usage_file = usage_file
        self._usage_records: List[UsageRecord] = []
        
        # Tier-based pricing multipliers
        self._tier_multipliers = {
            BillingTier.FREE: 0.0,        # Free tier (limited usage)
            BillingTier.BASIC: 1.0,       # Standard pricing
            BillingTier.PREMIUM: 0.8,     # 20% discount
            BillingTier.ENTERPRISE: 0.6   # 40% discount
        }
        
        # Load existing usage records if file specified
        if self.usage_file:
            self._load_usage_records()
    
    def calculate_cost(self, 
                      model_id: str, 
                      input_tokens: int, 
                      output_tokens: int) -> float:
        """
        Calculate cost for model usage.
        
        Args:
            model_id: ID of the model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        # Get model pricing from registry
        base_cost = ModelRegistry.estimate_cost(model_id, input_tokens, output_tokens)
        
        # Apply billing tier multiplier
        tier_multiplier = self._tier_multipliers.get(self.billing_tier, 1.0)
        final_cost = base_cost * tier_multiplier
        
        return final_cost
    
    def record_usage(self, 
                    model_id: str, 
                    input_tokens: int, 
                    output_tokens: int,
                    session_id: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> UsageRecord:
        """
        Record usage and calculate cost.
        
        Args:
            model_id: ID of the model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            session_id: Optional session identifier
            metadata: Additional metadata
            
        Returns:
            UsageRecord with cost calculated
        """
        cost = self.calculate_cost(model_id, input_tokens, output_tokens)
        
        record = UsageRecord(
            timestamp=datetime.now(),
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            session_id=session_id,
            metadata=metadata or {}
        )
        
        if self.track_usage:
            self._usage_records.append(record)
            
            # Persist to file if specified
            if self.usage_file:
                self._save_usage_records()
        
        return record
    
    def get_usage_summary(self, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         model_id: Optional[str] = None,
                         session_id: Optional[str] = None) -> CostSummary:
        """
        Get usage summary for a period or filter.
        
        Args:
            start_date: Start of period (default: beginning of records)
            end_date: End of period (default: now)
            model_id: Filter by specific model
            session_id: Filter by specific session
            
        Returns:
            CostSummary with aggregated data
        """
        # Filter records
        filtered_records = self._filter_records(
            start_date, end_date, model_id, session_id
        )
        
        if not filtered_records:
            return CostSummary(
                total_cost=0.0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_requests=0,
                cost_by_model={},
                period_start=start_date or datetime.now(),
                period_end=end_date or datetime.now(),
                average_cost_per_request=0.0
            )
        
        # Calculate totals
        total_cost = sum(record.cost for record in filtered_records)
        total_input_tokens = sum(record.input_tokens for record in filtered_records)
        total_output_tokens = sum(record.output_tokens for record in filtered_records)
        total_requests = len(filtered_records)
        
        # Calculate cost by model
        cost_by_model = {}
        for record in filtered_records:
            if record.model_id not in cost_by_model:
                cost_by_model[record.model_id] = 0.0
            cost_by_model[record.model_id] += record.cost
        
        # Find most/least expensive models
        most_expensive = max(cost_by_model.keys(), key=cost_by_model.get) if cost_by_model else None
        cheapest = min(cost_by_model.keys(), key=cost_by_model.get) if cost_by_model else None
        
        # Determine period
        period_start = start_date or min(record.timestamp for record in filtered_records)
        period_end = end_date or max(record.timestamp for record in filtered_records)
        
        return CostSummary(
            total_cost=total_cost,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_requests=total_requests,
            cost_by_model=cost_by_model,
            period_start=period_start,
            period_end=period_end,
            average_cost_per_request=total_cost / total_requests,
            most_expensive_model=most_expensive,
            cheapest_model=cheapest
        )
    
    def estimate_batch_cost(self, 
                           model_id: str, 
                           questions: List[str],
                           estimated_input_tokens_per_question: int = 100,
                           estimated_output_tokens_per_question: int = 50) -> Dict[str, Any]:
        """
        Estimate cost for a batch of questions.
        
        Args:
            model_id: ID of the model to use
            questions: List of questions
            estimated_input_tokens_per_question: Average input tokens per question
            estimated_output_tokens_per_question: Average output tokens per question
            
        Returns:
            Dictionary with cost estimates
        """
        num_questions = len(questions)
        total_input_tokens = num_questions * estimated_input_tokens_per_question
        total_output_tokens = num_questions * estimated_output_tokens_per_question
        
        total_cost = self.calculate_cost(model_id, total_input_tokens, total_output_tokens)
        cost_per_question = total_cost / num_questions if num_questions > 0 else 0.0
        
        # Get model config for additional info
        model_config = ModelRegistry.get_model_config(model_id)
        
        return {
            'model_id': model_id,
            'model_name': model_config.display_name if model_config else model_id,
            'num_questions': num_questions,
            'estimated_total_cost': total_cost,
            'cost_per_question': cost_per_question,
            'estimated_input_tokens': total_input_tokens,
            'estimated_output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens,
            'billing_tier': self.billing_tier,
            'tier_discount': (1 - self._tier_multipliers[self.billing_tier]) * 100
        }
    
    def get_cost_breakdown_by_model(self, 
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get detailed cost breakdown by model.
        
        Args:
            start_date: Start of period
            end_date: End of period
            
        Returns:
            List of model cost breakdowns
        """
        filtered_records = self._filter_records(start_date, end_date)
        
        # Group by model
        model_stats = {}
        for record in filtered_records:
            if record.model_id not in model_stats:
                model_stats[record.model_id] = {
                    'model_id': record.model_id,
                    'total_cost': 0.0,
                    'total_requests': 0,
                    'total_input_tokens': 0,
                    'total_output_tokens': 0,
                    'average_cost_per_request': 0.0,
                    'first_used': record.timestamp,
                    'last_used': record.timestamp
                }
            
            stats = model_stats[record.model_id]
            stats['total_cost'] += record.cost
            stats['total_requests'] += 1
            stats['total_input_tokens'] += record.input_tokens
            stats['total_output_tokens'] += record.output_tokens
            
            if record.timestamp < stats['first_used']:
                stats['first_used'] = record.timestamp
            if record.timestamp > stats['last_used']:
                stats['last_used'] = record.timestamp
        
        # Calculate averages and add model info
        breakdown = []
        for model_id, stats in model_stats.items():
            model_config = ModelRegistry.get_model_config(model_id)
            
            stats['average_cost_per_request'] = stats['total_cost'] / stats['total_requests']
            stats['model_name'] = model_config.display_name if model_config else model_id
            stats['provider'] = model_config.provider.value if model_config else 'unknown'
            
            breakdown.append(stats)
        
        # Sort by total cost (highest first)
        breakdown.sort(key=lambda x: x['total_cost'], reverse=True)
        
        return breakdown
    
    def get_daily_costs(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get daily cost breakdown for the last N days.
        
        Args:
            days: Number of days to include
            
        Returns:
            List of daily cost summaries
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        filtered_records = self._filter_records(start_date, end_date)
        
        # Group by date
        daily_costs = {}
        for record in filtered_records:
            date_key = record.timestamp.date()
            if date_key not in daily_costs:
                daily_costs[date_key] = {
                    'date': date_key,
                    'total_cost': 0.0,
                    'total_requests': 0,
                    'models_used': set()
                }
            
            daily_costs[date_key]['total_cost'] += record.cost
            daily_costs[date_key]['total_requests'] += 1
            daily_costs[date_key]['models_used'].add(record.model_id)
        
        # Convert to list and sort by date
        daily_list = []
        for date_key, stats in daily_costs.items():
            stats['models_used'] = list(stats['models_used'])
            stats['num_models'] = len(stats['models_used'])
            daily_list.append(stats)
        
        daily_list.sort(key=lambda x: x['date'])
        
        return daily_list
    
    def _filter_records(self, 
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       model_id: Optional[str] = None,
                       session_id: Optional[str] = None) -> List[UsageRecord]:
        """Filter usage records based on criteria."""
        filtered = self._usage_records
        
        if start_date:
            filtered = [r for r in filtered if r.timestamp >= start_date]
        
        if end_date:
            filtered = [r for r in filtered if r.timestamp <= end_date]
        
        if model_id:
            filtered = [r for r in filtered if r.model_id == model_id]
        
        if session_id:
            filtered = [r for r in filtered if r.session_id == session_id]
        
        return filtered
    
    def _save_usage_records(self):
        """Save usage records to file."""
        if not self.usage_file:
            return
        
        try:
            records_data = []
            for record in self._usage_records:
                records_data.append({
                    'timestamp': record.timestamp.isoformat(),
                    'model_id': record.model_id,
                    'input_tokens': record.input_tokens,
                    'output_tokens': record.output_tokens,
                    'cost': record.cost,
                    'session_id': record.session_id,
                    'metadata': record.metadata
                })
            
            Path(self.usage_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.usage_file, 'w') as f:
                json.dump(records_data, f, indent=2)
                
        except Exception as e:
            # Log error but don't fail the operation
            print(f"Warning: Failed to save usage records: {e}")
    
    def _load_usage_records(self):
        """Load usage records from file."""
        if not self.usage_file or not Path(self.usage_file).exists():
            return
        
        try:
            with open(self.usage_file, 'r') as f:
                records_data = json.load(f)
            
            for record_data in records_data:
                record = UsageRecord(
                    timestamp=datetime.fromisoformat(record_data['timestamp']),
                    model_id=record_data['model_id'],
                    input_tokens=record_data['input_tokens'],
                    output_tokens=record_data['output_tokens'],
                    cost=record_data['cost'],
                    session_id=record_data.get('session_id'),
                    metadata=record_data.get('metadata', {})
                )
                self._usage_records.append(record)
                
        except Exception as e:
            # Log error but don't fail initialization
            print(f"Warning: Failed to load usage records: {e}")
    
    def clear_usage_history(self):
        """Clear all usage records."""
        self._usage_records.clear()
        if self.usage_file and Path(self.usage_file).exists():
            Path(self.usage_file).unlink()
    
    def export_usage_data(self, output_file: str, format: str = 'json'):
        """
        Export usage data to file.
        
        Args:
            output_file: Output file path
            format: Export format ('json' or 'csv')
        """
        if format == 'json':
            self._export_json(output_file)
        elif format == 'csv':
            self._export_csv(output_file)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, output_file: str):
        """Export usage data as JSON."""
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'billing_tier': self.billing_tier,
            'total_records': len(self._usage_records),
            'records': []
        }
        
        for record in self._usage_records:
            data['records'].append({
                'timestamp': record.timestamp.isoformat(),
                'model_id': record.model_id,
                'input_tokens': record.input_tokens,
                'output_tokens': record.output_tokens,
                'cost': record.cost,
                'session_id': record.session_id,
                'metadata': record.metadata
            })
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _export_csv(self, output_file: str):
        """Export usage data as CSV."""
        import csv
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'timestamp', 'model_id', 'input_tokens', 'output_tokens', 
                'cost', 'session_id', 'metadata'
            ])
            
            # Data
            for record in self._usage_records:
                writer.writerow([
                    record.timestamp.isoformat(),
                    record.model_id,
                    record.input_tokens,
                    record.output_tokens,
                    record.cost,
                    record.session_id,
                    json.dumps(record.metadata) if record.metadata else ''
                ])