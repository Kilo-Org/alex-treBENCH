# Database Integration Setup

This document explains how to set up database integration for the alex-treBENCH frontend.

## Prerequisites

1. **PostgreSQL Database**: The frontend requires a PostgreSQL database. SQLite is not supported for the frontend due to Next.js limitations.

2. **Backend Data**: You need to have run benchmarks using the alex-treBENCH CLI tool to have data in the database.

## Setup Steps

### 1. Database Configuration

The frontend connects to the same PostgreSQL database used by the CLI tool.

#### Option A: Use Existing CLI Database

If you're already using PostgreSQL with the CLI tool:

```bash
# Copy your existing DATABASE_URL from the main project
cp ../.env frontend/.env.local
```

#### Option B: Set up New PostgreSQL Database

```bash
# Copy the example environment file
cp .env.local.example .env.local

# Edit .env.local with your PostgreSQL credentials
DATABASE_URL=postgresql://username:password@localhost:5432/alex_trebench
```

### 2. Install Dependencies

Dependencies are already installed if you ran `pnpm install` after the package.json updates.

If needed:
```bash
pnpm install pg @types/pg
```

### 3. Database Schema

The frontend uses the existing database schema from the CLI tool:

- `questions` - Jeopardy questions cache
- `benchmark_runs` - Benchmark execution records  
- `benchmark_results` - Individual question results
- `model_performance` - Aggregated performance metrics

### 4. Run the Frontend

```bash
pnpm dev
```

The frontend will be available at http://localhost:3000

## API Endpoints

The frontend provides these API endpoints:

- `GET /api/stats/summary` - Dashboard statistics
- `GET /api/leaderboard` - Model performance leaderboard

## Data Requirements

For the frontend to display meaningful data, you need:

1. **Questions Data**: Run `alex data init` to populate the questions table
2. **Benchmark Data**: Run at least one benchmark with `alex benchmark run`

Example CLI commands to populate data:

```bash
# Initialize questions database
alex data init

# Run a quick benchmark to generate performance data
alex benchmark run --models gpt-3.5-turbo --size quick

# Or run a larger benchmark for more comprehensive data
alex benchmark run --models gpt-4,claude-3-haiku,gpt-3.5-turbo --size small
```

## Troubleshooting

### Database Connection Issues

1. **Check DATABASE_URL**: Ensure it points to a valid PostgreSQL instance
2. **Verify Credentials**: Test connection with `psql` command
3. **Network Access**: Ensure PostgreSQL accepts connections from your host

### No Data Showing

1. **Check Tables**: Verify tables exist and contain data
2. **Run Benchmarks**: Execute CLI benchmarks to populate performance data
3. **Check API Responses**: Visit `/api/stats/summary` directly to see raw data

### Performance Issues

1. **Database Indexes**: Ensure proper indexing on frequently queried columns
2. **Connection Pool**: Adjust pool settings in `lib/database.ts` if needed
3. **Query Optimization**: Check slow query logs if performance degrades

## Development Notes

### Database Connection

- Uses connection pooling for performance
- Automatic connection management with cleanup
- Error handling and reconnection logic

### Type Safety

- Full TypeScript support with database types
- Interface definitions match SQLAlchemy models
- Type-safe query functions

### Error Handling

- Graceful degradation when database unavailable
- Fallback data for leaderboard in case of errors
- User-friendly error messages with retry options

## Production Deployment

For production deployment:

1. Set `NODE_ENV=production`
2. Use proper PostgreSQL connection string with SSL
3. Configure connection pool limits appropriately
4. Set up database monitoring and alerting
5. Implement backup strategies

## Security Considerations

- Never commit `.env.local` to version control
- Use environment variables for database credentials
- Enable SSL connections in production
- Implement proper access controls on database