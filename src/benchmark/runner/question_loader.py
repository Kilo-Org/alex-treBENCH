"""
Question Loader Module

Handles loading and sampling of questions for benchmarks.
"""

from typing import List, Dict, Any
import pandas as pd
import logging

from src.core.config import get_config
from src.core.database import get_db_session
from src.data.sampling import StatisticalSampler
from src.storage.repositories import QuestionRepository
from src.core.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class QuestionLoader:
    """Handles loading and sampling of questions for benchmarks."""
    
    def __init__(self, sampler: StatisticalSampler):
        """
        Initialize the question loader.
        
        Args:
            sampler: StatisticalSampler instance for sampling questions
        """
        self.sampler = sampler
        self.config = get_config()
    
    async def load_sample_questions(self, benchmark_id: int, config: 'BenchmarkConfig') -> List[Dict[str, Any]]:
        """
        Load and sample questions for the benchmark.
        
        Args:
            benchmark_id: ID of the benchmark run
            config: Benchmark configuration with sampling parameters
            
        Returns:
            List of sampled question dictionaries
        """
        try:
            logger.info(f"🔍 DEBUG: Starting load_sample_questions - sample_size={config.sample_size}, method={config.sampling_method}")
            
            # Get questions from database using repository
            logger.info("🔍 DEBUG: Opening database session...")
            with get_db_session() as session:
                logger.info("🔍 DEBUG: Creating QuestionRepository...")
                question_repo = QuestionRepository(session)
                
                # Get all questions as dataframe for sampling
                logger.info("🔍 DEBUG: Calling get_all_questions()...")
                all_questions = question_repo.get_all_questions()
                logger.info(f"🔍 DEBUG: Retrieved {len(all_questions) if all_questions else 0} questions from database")
                
                if not all_questions:
                    raise DatabaseError("No questions found in database. Please run data initialization first.")
                
                logger.info(f"Found {len(all_questions)} total questions in database")
                
                # Convert to DataFrame for sampling
                questions_data = []
                for q in all_questions:
                    questions_data.append({
                        'id': q.id,
                        'question_text': q.question_text,
                        'correct_answer': q.correct_answer,
                        'category': q.category,
                        'value': q.value or 400,  # Default value if None
                        'difficulty_level': q.difficulty_level or 'Medium',  # Default if None
                        'air_date': q.air_date,
                        'show_number': q.show_number,
                        'round': q.round
                    })
                
                df = pd.DataFrame(questions_data)
                
                # Use statistical sampler to get representative sample
                if config.sampling_method == "stratified":
                    sampled_df = self.sampler.stratified_sample(
                        df=df,
                        sample_size=config.sample_size,
                        stratify_columns=config.stratify_columns,
                        seed=config.sampling_seed
                    )
                else:
                    # Fall back to simple random sampling
                    sampled_df = df.sample(n=min(config.sample_size, len(df)), 
                                         random_state=config.sampling_seed)
                
                # Convert back to list of dicts with proper type handling
                sample_records = sampled_df.to_dict('records')
                sample_questions: List[Dict[str, Any]] = []
                for record in sample_records:
                    # Convert keys to strings and add to list
                    str_record: Dict[str, Any] = {str(k): v for k, v in record.items()}
                    sample_questions.append(str_record)
                
                logger.info(f"Sampled {len(sample_questions)} questions for benchmark")
                
                # Log sampling distribution
                if 'category' in sampled_df.columns:
                    category_counts = sampled_df['category'].value_counts()
                    logger.debug(f"Category distribution: {category_counts.head(10).to_dict()}")
                
                return sample_questions
                
        except Exception as e:
            logger.error(f"Failed to load sample questions: {str(e)}")
            raise DatabaseError(f"Failed to load sample questions: {str(e)}")