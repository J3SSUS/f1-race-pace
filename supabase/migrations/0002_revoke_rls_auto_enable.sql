-- `rls_auto_enable()` is created by Supabase's "Enable automatic RLS" option and is a
-- SECURITY DEFINER function. Keep it out of the Data API: the event trigger still fires.
REVOKE EXECUTE ON FUNCTION public.rls_auto_enable() FROM anon, authenticated;
