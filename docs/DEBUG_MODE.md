# Debugging alex-treBENCH

## Common Issues and Solutions

### Database Connection Errors

#### Issue: libSQL/Turso Integration Problems

**Symptoms:**
- `RuntimeWarning: coroutine 'HranaClient.execute' was never awaited`
- `RuntimeWarning: coroutine 'HranaClient.close' was never awaited`
- `TypeError: Connection() got an unexpected keyword argument '_libsql_url'`
- `RuntimeError: asyncio.run() cannot be called from a running event loop`

**Root Cause:**
The system provides experimental support for libSQL/Turso databases (URLs starting with `libsql://`). However, full integration requires a custom SQLAlchemy dialect. As a temporary measure, the system falls back to a local SQLite database (`database/turso_local_cache.db`) for SQLAlchemy compatibility.

**Problems Encountered:**
1. **Async/Sync Mismatch:** libSQL client is asynchronous, but database engine creation occurs in synchronous contexts (e.g., health checks), leading to unawaited coroutines.
2. **Connect Args Pollution:** libSQL-specific parameters were passed to SQLite's connection constructor, causing TypeError.
3. **Nested Event Loops:** Attempts to use `asyncio.run()` within already-running event loops (common in async CLI commands) caused RuntimeErrors.

**Resolution:**
The system now:
- Skips the libSQL connection test to avoid async/sync conflicts.
- Uses clean connect_args for the SQLite fallback engine.
- Warns about experimental status and local SQLite usage.
- Maintains libSQL URL metadata for potential future use.

**Configuration:**
- **libSQL URL:** Must include `authToken` parameter or set `TURSO_AUTH_TOKEN` environment variable.
- **Fallback:** Automatically creates `database/turso_local_cache.db` for local operations.
- **Production Note:** For full libSQL support, implement a custom SQLAlchemy dialect or use synchronous libSQL alternatives.

**Verification:**
Run `alex health --check-db` to confirm connection works. Expected output:
```
✓ Database connection: OK
  Type: libSQL/Turso
  URL: libsql://your-database.turso.io (Auth: ✓)
```

#### Issue: Missing Auth Token

**Symptoms:**
- `Turso auth token is required for libSQL connections`
- Database connection test failed

**Solution:**
Add `authToken=your_token` to the libSQL URL in `.env`:
```
DATABASE_URL=libsql://your-database.turso.io?authToken=your_auth_token
```
Or set environment variable:
```
export TURSO_AUTH_TOKEN=your_auth_token
```

### Logging Configuration

To debug database issues, enable debug logging in `config/default.yaml`:
```yaml
logging:
  level: "DEBUG"
  debug:
    enabled: true
```

This will show detailed engine creation logs, including the runtime database URL and configuration path taken.

### Testing Database Fixes

After applying fixes:
1. Run `alex health -v --check-db` to verify connection.
2. Check for the warning about experimental libSQL support.
3. Verify no RuntimeWarnings or TypeErrors appear.
4. Confirm the local SQLite cache is created at `database/turso_local_cache.db`.

### Future Improvements

- Implement custom SQLAlchemy dialect for full libSQL support.
- Add async-aware connection testing.
- Support libSQL-specific features (replication, etc.) in production.
- Add configuration option to disable libSQL fallback.

## Other Debugging Tips

- Use `alex health -v` for full system health check.
- Check logs in `logs/benchmark.log` for detailed error traces.
- Verify environment variables with `printenv | grep -E "(DATABASE_URL|TURSO_AUTH_TOKEN)"`.
- Test database initialization with `alex data init --dry-run`.

For persistent issues, check the [Technical Specification](TECHNICAL_SPEC.md) for architecture details or report issues with full stack traces.