import { NextResponse } from 'next/server';
import { executeQuerySingle, executeQueryAll, StatsResponse, formatNumber } from '@/lib/database';

export async function GET() {
  try {
    // Query 1: Total questions count from questions table
    const totalQuestionsResult = await executeQuerySingle<{ count: string }>(
      'SELECT COUNT(*) as count FROM questions'
    );
    const totalQuestions = parseInt(totalQuestionsResult?.count || '0');

    // Query 2: Count distinct models tested from model_performance table
    // This is more reliable than parsing JSON from benchmark_runs.models_tested
    const modelsTestedResult = await executeQuerySingle<{ count: string }>(
      'SELECT COUNT(DISTINCT model_name) as count FROM model_performance'
    );
    const modelsTested = parseInt(modelsTestedResult?.count || '0');

    // Query 3: Total questions answered from benchmark_results
    const questionsAnsweredResult = await executeQuerySingle<{ count: string }>(
      'SELECT COUNT(*) as count FROM benchmark_results'
    );
    const questionsAnswered = parseInt(questionsAnsweredResult?.count || '0');

    // Query 4: Best accuracy from model_performance table
    const bestAccuracyResult = await executeQuerySingle<{ accuracy_rate: number }>(
      'SELECT MAX(accuracy_rate) as accuracy_rate FROM model_performance'
    );
    const bestAccuracy = bestAccuracyResult?.accuracy_rate || 0;

    // Format the response
    const stats: StatsResponse = {
      totalQuestions,
      modelsTested,
      questionsAnswered,
      bestAccuracy: Math.round(bestAccuracy * 100 * 100) / 100, // Convert to percentage with 2 decimal places
    };

    return NextResponse.json(stats);

  } catch (error) {
    console.error('Error fetching stats summary:', error);
    
    // Return error response
    return NextResponse.json(
      { 
        error: 'Failed to fetch statistics',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}