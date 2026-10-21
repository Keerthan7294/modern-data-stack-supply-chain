-- Supply Chain Analytics DB Initialization Script
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Grant permissions to postgres user
GRANT ALL ON SCHEMA raw TO postgres;
GRANT ALL ON SCHEMA analytics TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA raw GRANT ALL ON TABLES TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics GRANT ALL ON TABLES TO postgres;
