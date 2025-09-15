#!/usr/bin/env node

// Simple integration test for alex-treBENCH frontend database integration
// This script tests the API endpoints without requiring a running database

console.log('🧪 Testing alex-treBENCH Frontend Database Integration\n');

const tests = [
  {
    name: 'Stats Summary API Structure',
    test: async () => {
      // Test that the API route file exists and has proper structure
      const fs = require('fs');
      const path = require('path');
      
      const apiPath = path.join(__dirname, 'src/app/api/stats/summary/route.ts');
      const leaderboardPath = path.join(__dirname, 'src/app/api/leaderboard/route.ts');
      
      if (!fs.existsSync(apiPath)) {
        throw new Error('Stats summary API route file not found');
      }
      
      if (!fs.existsSync(leaderboardPath)) {
        throw new Error('Leaderboard API route file not found');
      }
      
      const statsContent = fs.readFileSync(apiPath, 'utf8');
      const leaderboardContent = fs.readFileSync(leaderboardPath, 'utf8');
      
      // Check for required exports and functions
      if (!statsContent.includes('export async function GET()')) {
        throw new Error('Stats API missing GET export');
      }
      
      if (!leaderboardContent.includes('export async function GET()')) {
        throw new Error('Leaderboard API missing GET export');
      }
      
      console.log('  ✅ API route files exist and have proper structure');
    }
  },
  
  {
    name: 'Database Connection Configuration',
    test: async () => {
      const fs = require('fs');
      const path = require('path');
      
      const dbPath = path.join(__dirname, 'src/lib/database.ts');
      const nextConfigPath = path.join(__dirname, 'next.config.ts');
      
      if (!fs.existsSync(dbPath)) {
        throw new Error('Database utility file not found');
      }
      
      const dbContent = fs.readFileSync(dbPath, 'utf8');
      const nextConfigContent = fs.readFileSync(nextConfigPath, 'utf8');
      
      // Check for required database functions
      if (!dbContent.includes('executeQuery')) {
        throw new Error('Database utility missing executeQuery function');
      }
      
      if (!dbContent.includes('initializeDatabase')) {
        throw new Error('Database utility missing initializeDatabase function');
      }
      
      // Check Next.js config
      if (!nextConfigContent.includes('DATABASE_URL')) {
        throw new Error('Next.js config missing DATABASE_URL environment variable');
      }
      
      console.log('  ✅ Database configuration is properly set up');
    }
  },
  
  {
    name: 'Component Integration',
    test: async () => {
      const fs = require('fs');
      const path = require('path');
      
      const statsPath = path.join(__dirname, 'src/app/components/HomeStats.tsx');
      const leaderboardPath = path.join(__dirname, 'src/app/components/Leaderboard.tsx');
      
      const statsContent = fs.readFileSync(statsPath, 'utf8');
      const leaderboardContent = fs.readFileSync(leaderboardPath, 'utf8');
      
      // Check that components use real data (no hardcoded mock data)
      if (!statsContent.includes('fetch(\'/api/stats/summary\')')) {
        throw new Error('HomeStats component not using real API data');
      }
      
      if (!leaderboardContent.includes('fetch(\'/api/leaderboard\')')) {
        throw new Error('Leaderboard component not using real API data');
      }
      
      // Check for loading states
      if (!statsContent.includes('loading') || !leaderboardContent.includes('loading')) {
        throw new Error('Components missing loading states');
      }
      
      // Check for error handling
      if (!statsContent.includes('error') || !leaderboardContent.includes('error')) {
        throw new Error('Components missing error handling');
      }
      
      console.log('  ✅ Components properly integrated with API and have loading/error states');
    }
  },
  
  {
    name: 'TypeScript Configuration',
    test: async () => {
      const { execSync } = require('child_process');
      
      try {
        // Check TypeScript compilation
        execSync('pnpm tsc --noEmit', { 
          stdio: 'pipe',
          cwd: __dirname 
        });
        console.log('  ✅ TypeScript compilation successful');
      } catch (error) {
        throw new Error(`TypeScript compilation failed: ${error.message}`);
      }
    }
  },
  
  {
    name: 'Build Process',
    test: async () => {
      const { execSync } = require('child_process');
      
      try {
        // Test Next.js build
        execSync('pnpm build', { 
          stdio: 'pipe',
          cwd: __dirname 
        });
        console.log('  ✅ Next.js build successful');
      } catch (error) {
        throw new Error(`Next.js build failed: ${error.message}`);
      }
    }
  }
];

async function runTests() {
  let passed = 0;
  let failed = 0;
  
  for (const test of tests) {
    try {
      console.log(`🧪 ${test.name}:`);
      await test.test();
      passed++;
    } catch (error) {
      console.log(`  ❌ ${error.message}`);
      failed++;
    }
    console.log('');
  }
  
  console.log(`📊 Test Results:`);
  console.log(`  ✅ Passed: ${passed}`);
  console.log(`  ❌ Failed: ${failed}`);
  console.log(`  📈 Success Rate: ${((passed / (passed + failed)) * 100).toFixed(1)}%`);
  
  if (failed > 0) {
    console.log('\n⚠️  Some tests failed. Please check the issues above.');
    process.exit(1);
  } else {
    console.log('\n🎉 All integration tests passed!');
    console.log('\n📝 Next Steps:');
    console.log('  1. Set up your DATABASE_URL in .env.local');
    console.log('  2. Ensure you have benchmark data in your database');
    console.log('  3. Run `pnpm dev` to start the development server');
    console.log('  4. Visit http://localhost:3000 to see real data');
  }
}

runTests().catch(error => {
  console.error('❌ Test runner error:', error.message);
  process.exit(1);
});