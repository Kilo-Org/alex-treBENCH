#!/usr/bin/env node

// Quick database inspection for debugging models tested count issue
const { Pool } = require('pg');

async function testDatabase() {
  console.log('🔍 Testing PostgreSQL Database Connection and Data\n');
  
  // Get database URL from environment
  const databaseUrl = process.env.DATABASE_URL;
  
  if (!databaseUrl) {
    console.error('❌ DATABASE_URL not found in environment');
    process.exit(1);
  }
  
  console.log('📡 Connecting to:', databaseUrl.split('@')[1]?.split('/')[0] || 'database');
  
  const pool = new Pool({
    connectionString: databaseUrl,
    ssl: { rejectUnauthorized: false }
  });

  try {
    // Test connection
    console.log('✅ Database connection established');
    
    // Check what tables exist
    console.log('\n📋 Available Tables:');
    const tablesResult = await pool.query(`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public'
      ORDER BY table_name
    `);
    
    if (tablesResult.rows.length === 0) {
      console.log('  ❌ No tables found in database');
      return;
    }
    
    tablesResult.rows.forEach(row => {
      console.log(`  - ${row.table_name}`);
    });
    
    // Check each key table for data
    const keyTables = ['questions', 'benchmark_runs', 'benchmark_results', 'model_performance'];
    
    console.log('\n📊 Table Data Summary:');
    for (const tableName of keyTables) {
      try {
        const countResult = await pool.query(`SELECT COUNT(*) as count FROM ${tableName}`);
        console.log(`  ${tableName}: ${countResult.rows[0].count} rows`);
        
        // For benchmark_runs, also show models_tested field structure
        if (tableName === 'benchmark_runs' && parseInt(countResult.rows[0].count) > 0) {
          const sampleResult = await pool.query(`
            SELECT id, models_tested, status 
            FROM ${tableName} 
            LIMIT 3
          `);
          console.log(`    Sample models_tested data:`);
          sampleResult.rows.forEach(row => {
            console.log(`      ID ${row.id}: "${row.models_tested}" (status: ${row.status})`);
          });
        }
        
        // For model_performance, show unique models
        if (tableName === 'model_performance' && parseInt(countResult.rows[0].count) > 0) {
          const modelsResult = await pool.query(`
            SELECT DISTINCT model_name 
            FROM ${tableName} 
            ORDER BY model_name 
            LIMIT 5
          `);
          console.log(`    Unique models: ${modelsResult.rows.map(r => r.model_name).join(', ')}`);
        }
        
        // For benchmark_results, show unique models  
        if (tableName === 'benchmark_results' && parseInt(countResult.rows[0].count) > 0) {
          const modelsResult = await pool.query(`
            SELECT DISTINCT model_name 
            FROM ${tableName} 
            ORDER BY model_name 
            LIMIT 5
          `);
          console.log(`    Unique models: ${modelsResult.rows.map(r => r.model_name).join(', ')}`);
        }
        
      } catch (error) {
        console.log(`  ${tableName}: ❌ Table not found or error (${error.message})`);
      }
    }
    
    // Test the current problematic query
    console.log('\n🐛 Testing Current Problematic Query:');
    try {
      const currentQueryResult = await pool.query(`
        SELECT DISTINCT models_tested 
        FROM benchmark_runs 
        WHERE models_tested IS NOT NULL
      `);
      console.log(`  Query returned ${currentQueryResult.rows.length} rows`);
      currentQueryResult.rows.forEach((row, index) => {
        console.log(`  Row ${index + 1}: "${row.models_tested}"`);
        console.log(`  Is array? ${Array.isArray(row.models_tested)}`);
        console.log(`  Type: ${typeof row.models_tested}`);
      });
    } catch (error) {
      console.log(`  ❌ Current query failed: ${error.message}`);
    }
    
    // Test better alternative queries
    console.log('\n✨ Testing Better Alternative Queries:');
    
    // Option 1: From model_performance
    try {
      const perfResult = await pool.query(`
        SELECT COUNT(DISTINCT model_name) as count 
        FROM model_performance
      `);
      console.log(`  model_performance distinct models: ${perfResult.rows[0].count}`);
    } catch (error) {
      console.log(`  ❌ model_performance query failed: ${error.message}`);
    }
    
    // Option 2: From benchmark_results
    try {
      const resultsResult = await pool.query(`
        SELECT COUNT(DISTINCT model_name) as count 
        FROM benchmark_results
      `);
      console.log(`  benchmark_results distinct models: ${resultsResult.rows[0].count}`);
    } catch (error) {
      console.log(`  ❌ benchmark_results query failed: ${error.message}`);
    }
    
  } catch (error) {
    console.error('❌ Database test failed:', error.message);
  } finally {
    await pool.end();
    console.log('\n✅ Database connection closed');
  }
}

// Load environment variables
require('dotenv').config({ path: '.env.local' });

testDatabase().catch(error => {
  console.error('❌ Test failed:', error.message);
  process.exit(1);
});