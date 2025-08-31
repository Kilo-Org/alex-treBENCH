"""
Tests for Cost Calculator
"""

import pytest
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock

from src.models.cost_calculator import (
    CostCalculator, BillingTier, UsageRecord, CostSummary
)


class TestCostCalculator:
    """Test cases for CostCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = CostCalculator(
            billing_tier=BillingTier.BASIC,
            track_usage=True
        )
        
        # Sample usage records for testing
        self.sample_records = [
            UsageRecord(
                timestamp=datetime(2024, 1, 1, 10, 0),
                model_id="openai/gpt-3.5-turbo",
                input_tokens=100,
                output_tokens=50,
                cost=0.001,
                session_id="session1"
            ),
            UsageRecord(
                timestamp=datetime(2024, 1, 1, 11, 0),
                model_id="anthropic/claude-3-haiku",
                input_tokens=200,
                output_tokens=75,
                cost=0.002,
                session_id="session1"
            ),
            UsageRecord(
                timestamp=datetime(2024, 1, 2, 10, 0),
                model_id="openai/gpt-3.5-turbo",
                input_tokens=150,
                output_tokens=100,
                cost=0.0015,
                session_id="session2"
            )
        ]
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        calculator = CostCalculator()
        
        assert calculator.billing_tier == BillingTier.BASIC
        assert calculator.track_usage == True
        assert calculator.usage_file is None
        assert calculator._usage_records == []
    
    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        calculator = CostCalculator(
            billing_tier=BillingTier.PREMIUM,
            track_usage=False,
            usage_file="test.json"
        )
        
        assert calculator.billing_tier == BillingTier.PREMIUM
        assert calculator.track_usage == False
        assert calculator.usage_file == "test.json"
    
    def test_calculate_cost_basic_tier(self):
        """Test cost calculation for basic tier."""
        cost = self.calculator.calculate_cost(
            "openai/gpt-3.5-turbo",
            input_tokens=1000,
            output_tokens=500
        )
        
        # Basic tier should have 1.0 multiplier (no discount)
        # gpt-3.5-turbo: $0.5 input, $1.5 output per 1M tokens
        expected_cost = (1000 / 1_000_000) * 0.5 + (500 / 1_000_000) * 1.5
        assert abs(cost - expected_cost) < 0.0001
    
    def test_calculate_cost_premium_tier(self):
        """Test cost calculation for premium tier (with discount)."""
        premium_calculator = CostCalculator(billing_tier=BillingTier.PREMIUM)
        
        cost = premium_calculator.calculate_cost(
            "openai/gpt-3.5-turbo",
            input_tokens=1000,
            output_tokens=500
        )
        
        # Premium tier should have 0.8 multiplier (20% discount)
        base_cost = (1000 / 1_000_000) * 0.5 + (500 / 1_000_000) * 1.5
        expected_cost = base_cost * 0.8
        assert abs(cost - expected_cost) < 0.0001
    
    def test_calculate_cost_free_tier(self):
        """Test cost calculation for free tier."""
        free_calculator = CostCalculator(billing_tier=BillingTier.FREE)
        
        cost = free_calculator.calculate_cost(
            "openai/gpt-3.5-turbo",
            input_tokens=1000,
            output_tokens=500
        )
        
        # Free tier should have 0.0 multiplier
        assert cost == 0.0
    
    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model."""
        cost = self.calculator.calculate_cost(
            "unknown/model",
            input_tokens=1000,
            output_tokens=500
        )
        
        assert cost == 0.0  # Should return 0 for unknown models
    
    def test_record_usage_with_tracking(self):
        """Test recording usage when tracking is enabled."""
        record = self.calculator.record_usage(
            model_id="openai/gpt-3.5-turbo",
            input_tokens=100,
            output_tokens=50,
            session_id="test-session",
            metadata={"test": "value"}
        )
        
        assert isinstance(record, UsageRecord)
        assert record.model_id == "openai/gpt-3.5-turbo"
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.cost > 0
        assert record.session_id == "test-session"
        assert record.metadata == {"test": "value"}
        assert isinstance(record.timestamp, datetime)
        
        # Should be added to records
        assert len(self.calculator._usage_records) == 1
        assert self.calculator._usage_records[0] is record
    
    def test_record_usage_without_tracking(self):
        """Test recording usage when tracking is disabled."""
        no_track_calculator = CostCalculator(track_usage=False)
        
        record = no_track_calculator.record_usage(
            model_id="openai/gpt-3.5-turbo",
            input_tokens=100,
            output_tokens=50
        )
        
        assert isinstance(record, UsageRecord)
        assert record.cost > 0
        
        # Should not be added to records
        assert len(no_track_calculator._usage_records) == 0
    
    def test_get_usage_summary_empty(self):
        """Test usage summary with no records."""
        summary = self.calculator.get_usage_summary()
        
        assert isinstance(summary, CostSummary)
        assert summary.total_cost == 0.0
        assert summary.total_input_tokens == 0
        assert summary.total_output_tokens == 0
        assert summary.total_requests == 0
        assert summary.cost_by_model == {}
        assert summary.average_cost_per_request == 0.0
    
    def test_get_usage_summary_with_data(self):
        """Test usage summary with sample data."""
        # Add sample records
        self.calculator._usage_records = self.sample_records
        
        summary = self.calculator.get_usage_summary()
        
        assert summary.total_cost == 0.0045  # Sum of all costs
        assert summary.total_input_tokens == 450  # Sum of input tokens
        assert summary.total_output_tokens == 225  # Sum of output tokens
        assert summary.total_requests == 3
        assert summary.average_cost_per_request == 0.0045 / 3
        
        # Check cost by model
        assert "openai/gpt-3.5-turbo" in summary.cost_by_model
        assert "anthropic/claude-3-haiku" in summary.cost_by_model
        assert summary.cost_by_model["openai/gpt-3.5-turbo"] == 0.0025  # Two records
        assert summary.cost_by_model["anthropic/claude-3-haiku"] == 0.002  # One record
        
        assert summary.most_expensive_model == "openai/gpt-3.5-turbo"
        assert summary.cheapest_model == "anthropic/claude-3-haiku"
    
    def test_get_usage_summary_with_filters(self):
        """Test usage summary with date and model filters."""
        self.calculator._usage_records = self.sample_records
        
        # Filter by date
        summary = self.calculator.get_usage_summary(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1, 23, 59)
        )
        
        assert summary.total_requests == 2  # Only Jan 1st records
        assert summary.total_cost == 0.003  # Only Jan 1st costs
        
        # Filter by model
        summary = self.calculator.get_usage_summary(
            model_id="openai/gpt-3.5-turbo"
        )
        
        assert summary.total_requests == 2  # Only GPT-3.5 records
        assert summary.total_cost == 0.0025  # Only GPT-3.5 costs
        assert "anthropic/claude-3-haiku" not in summary.cost_by_model
        
        # Filter by session
        summary = self.calculator.get_usage_summary(
            session_id="session1"
        )
        
        assert summary.total_requests == 2  # Only session1 records
        assert summary.total_cost == 0.003
    
    def test_estimate_batch_cost(self):
        """Test batch cost estimation."""
        questions = ["Question 1", "Question 2", "Question 3"]
        
        estimate = self.calculator.estimate_batch_cost(
            model_id="openai/gpt-3.5-turbo",
            questions=questions,
            estimated_input_tokens_per_question=100,
            estimated_output_tokens_per_question=50
        )
        
        assert isinstance(estimate, dict)
        assert estimate["model_id"] == "openai/gpt-3.5-turbo"
        assert estimate["num_questions"] == 3
        assert estimate["estimated_input_tokens"] == 300
        assert estimate["estimated_output_tokens"] == 150
        assert estimate["total_tokens"] == 450
        assert estimate["estimated_total_cost"] > 0
        assert estimate["cost_per_question"] == estimate["estimated_total_cost"] / 3
        assert estimate["billing_tier"] == BillingTier.BASIC
    
    def test_get_cost_breakdown_by_model(self):
        """Test getting cost breakdown by model."""
        self.calculator._usage_records = self.sample_records
        
        breakdown = self.calculator.get_cost_breakdown_by_model()
        
        assert isinstance(breakdown, list)
        assert len(breakdown) == 2  # Two different models
        
        # Should be sorted by total cost (highest first)
        assert breakdown[0]["total_cost"] >= breakdown[1]["total_cost"]
        
        # Check structure
        for model_breakdown in breakdown:
            assert "model_id" in model_breakdown
            assert "model_name" in model_breakdown
            assert "provider" in model_breakdown
            assert "total_cost" in model_breakdown
            assert "total_requests" in model_breakdown
            assert "total_input_tokens" in model_breakdown
            assert "total_output_tokens" in model_breakdown
            assert "average_cost_per_request" in model_breakdown
            assert "first_used" in model_breakdown
            assert "last_used" in model_breakdown
    
    def test_get_daily_costs(self):
        """Test getting daily cost breakdown."""
        self.calculator._usage_records = self.sample_records
        
        daily_costs = self.calculator.get_daily_costs(days=30)
        
        assert isinstance(daily_costs, list)
        assert len(daily_costs) == 2  # Two different dates in sample data
        
        # Should be sorted by date
        assert daily_costs[0]["date"] <= daily_costs[1]["date"]
        
        # Check structure
        for daily_cost in daily_costs:
            assert "date" in daily_cost
            assert "total_cost" in daily_cost
            assert "total_requests" in daily_cost
            assert "models_used" in daily_cost
            assert "num_models" in daily_cost
            assert isinstance(daily_cost["models_used"], list)
    
    def test_clear_usage_history(self):
        """Test clearing usage history."""
        self.calculator._usage_records = self.sample_records.copy()
        
        assert len(self.calculator._usage_records) == 3
        
        self.calculator.clear_usage_history()
        
        assert len(self.calculator._usage_records) == 0
    
    def test_save_and_load_usage_records(self):
        """Test saving and loading usage records to/from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            usage_file = f.name
        
        try:
            # Create calculator with usage file
            calculator = CostCalculator(usage_file=usage_file, track_usage=True)
            
            # Add some records
            calculator.record_usage("openai/gpt-3.5-turbo", 100, 50)
            calculator.record_usage("anthropic/claude-3-haiku", 200, 75)
            
            assert len(calculator._usage_records) == 2
            
            # Create new calculator with same file - should load records
            calculator2 = CostCalculator(usage_file=usage_file, track_usage=True)
            
            assert len(calculator2._usage_records) == 2
            assert calculator2._usage_records[0].model_id == "openai/gpt-3.5-turbo"
            assert calculator2._usage_records[1].model_id == "anthropic/claude-3-haiku"
            
        finally:
            # Cleanup
            Path(usage_file).unlink(missing_ok=True)
    
    def test_export_usage_data_json(self):
        """Test exporting usage data as JSON."""
        self.calculator._usage_records = self.sample_records
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name
        
        try:
            self.calculator.export_usage_data(output_file, format='json')
            
            # Verify file was created and has correct structure
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            assert "export_timestamp" in data
            assert "billing_tier" in data
            assert "total_records" in data
            assert "records" in data
            assert data["total_records"] == 3
            assert len(data["records"]) == 3
            
        finally:
            Path(output_file).unlink(missing_ok=True)
    
    def test_export_usage_data_csv(self):
        """Test exporting usage data as CSV."""
        self.calculator._usage_records = self.sample_records
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = f.name
        
        try:
            self.calculator.export_usage_data(output_file, format='csv')
            
            # Verify file was created
            assert Path(output_file).exists()
            
            # Check basic CSV structure
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 4  # Header + 3 data rows
            assert "timestamp,model_id,input_tokens,output_tokens,cost,session_id,metadata" in lines[0]
            
        finally:
            Path(output_file).unlink(missing_ok=True)
    
    def test_export_usage_data_invalid_format(self):
        """Test exporting with invalid format raises error."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            self.calculator.export_usage_data("output.txt", format='invalid')
    
    def test_tier_multipliers(self):
        """Test that billing tier multipliers are correctly applied."""
        test_cases = [
            (BillingTier.FREE, 0.0),
            (BillingTier.BASIC, 1.0),
            (BillingTier.PREMIUM, 0.8),
            (BillingTier.ENTERPRISE, 0.6)
        ]
        
        for tier, expected_multiplier in test_cases:
            calculator = CostCalculator(billing_tier=tier)
            
            base_cost = 1.0  # $1 base cost for easy calculation
            with patch('src.models.model_registry.ModelRegistry.estimate_cost', return_value=base_cost):
                actual_cost = calculator.calculate_cost("test/model", 1000, 500)
                expected_cost = base_cost * expected_multiplier
                
                assert abs(actual_cost - expected_cost) < 0.0001


class TestUsageRecord:
    """Test cases for UsageRecord class."""
    
    def test_usage_record_creation(self):
        """Test creating a UsageRecord."""
        timestamp = datetime.now()
        
        record = UsageRecord(
            timestamp=timestamp,
            model_id="openai/gpt-3.5-turbo",
            input_tokens=100,
            output_tokens=50,
            cost=0.001,
            session_id="test-session",
            metadata={"key": "value"}
        )
        
        assert record.timestamp == timestamp
        assert record.model_id == "openai/gpt-3.5-turbo"
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.cost == 0.001
        assert record.session_id == "test-session"
        assert record.metadata == {"key": "value"}
    
    def test_usage_record_defaults(self):
        """Test UsageRecord with default values."""
        record = UsageRecord(
            timestamp=datetime.now(),
            model_id="test/model",
            input_tokens=100,
            output_tokens=50,
            cost=0.001
        )
        
        assert record.session_id is None
        assert record.metadata == {}  # Should default to empty dict


class TestCostSummary:
    """Test cases for CostSummary class."""
    
    def test_cost_summary_creation(self):
        """Test creating a CostSummary."""
        start_time = datetime(2024, 1, 1)
        end_time = datetime(2024, 1, 31)
        
        summary = CostSummary(
            total_cost=10.50,
            total_input_tokens=5000,
            total_output_tokens=2500,
            total_requests=100,
            cost_by_model={"model1": 6.0, "model2": 4.5},
            period_start=start_time,
            period_end=end_time,
            average_cost_per_request=0.105,
            most_expensive_model="model1",
            cheapest_model="model2"
        )
        
        assert summary.total_cost == 10.50
        assert summary.total_input_tokens == 5000
        assert summary.total_output_tokens == 2500
        assert summary.total_requests == 100
        assert summary.cost_by_model == {"model1": 6.0, "model2": 4.5}
        assert summary.period_start == start_time
        assert summary.period_end == end_time
        assert summary.average_cost_per_request == 0.105
        assert summary.most_expensive_model == "model1"
        assert summary.cheapest_model == "model2"


class TestBillingTier:
    """Test cases for BillingTier enum."""
    
    def test_billing_tier_values(self):
        """Test that billing tier values are correct."""
        assert BillingTier.FREE == "free"
        assert BillingTier.BASIC == "basic"
        assert BillingTier.PREMIUM == "premium"
        assert BillingTier.ENTERPRISE == "enterprise"
    
    def test_billing_tier_membership(self):
        """Test billing tier enum membership."""
        assert "free" in BillingTier
        assert "basic" in BillingTier
        assert "premium" in BillingTier
        assert "enterprise" in BillingTier
        assert "invalid" not in BillingTier


def test_integration_cost_tracking():
    """Integration test for complete cost tracking workflow."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        usage_file = f.name
    
    try:
        # Create calculator and simulate usage
        calculator = CostCalculator(
            billing_tier=BillingTier.PREMIUM,
            usage_file=usage_file,
            track_usage=True
        )
        
        # Record some usage
        calculator.record_usage("openai/gpt-3.5-turbo", 1000, 500, session_id="session1")
        calculator.record_usage("anthropic/claude-3-haiku", 2000, 1000, session_id="session1")
        calculator.record_usage("openai/gpt-4", 500, 250, session_id="session2")
        
        # Get summary
        summary = calculator.get_usage_summary()
        
        assert summary.total_requests == 3
        assert summary.total_cost > 0
        assert len(summary.cost_by_model) == 3
        
        # Test filtering
        session1_summary = calculator.get_usage_summary(session_id="session1")
        assert session1_summary.total_requests == 2
        
        # Test cost breakdown
        breakdown = calculator.get_cost_breakdown_by_model()
        assert len(breakdown) == 3
        assert all("model_name" in b for b in breakdown)
        
        # Test batch estimation
        estimate = calculator.estimate_batch_cost(
            "openai/gpt-3.5-turbo",
            ["q1", "q2", "q3"],
            100, 50
        )
        assert estimate["num_questions"] == 3
        assert estimate["billing_tier"] == BillingTier.PREMIUM
        assert estimate["tier_discount"] == 20.0  # 20% discount for premium
        
        # Verify persistence
        calculator2 = CostCalculator(usage_file=usage_file)
        assert len(calculator2._usage_records) == 3
        
    finally:
        Path(usage_file).unlink(missing_ok=True)