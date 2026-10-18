-- 0002 revoked from anon and authenticated only, but they inherit EXECUTE from PUBLIC
-- (the default for functions), so the grant stayed. Revoke it from PUBLIC too.
REVOKE EXECUTE ON FUNCTION public.rls_auto_enable() FROM PUBLIC, anon, authenticated;
