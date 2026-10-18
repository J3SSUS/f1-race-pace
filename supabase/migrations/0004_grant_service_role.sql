-- The project was created with "Automatically expose new tables" off, so service_role got no
-- table privileges. It bypasses RLS but still needs GRANTs: the pipeline reads and writes
-- through it. `anon` is untouched (its grants are set in 0001).
GRANT ALL ON ALL TABLES IN SCHEMA public TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO service_role;
