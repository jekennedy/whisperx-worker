# Only run if setup version changed or venv missing
NEED_BOOTSTRAP=0
[ ! -d .venv ] && NEED_BOOTSTRAP=1
if [ -f .codex_setup_version ]; then
  CURR=3                                  # <- keep in sync with setup.sh
  LAST=$(cat .codex_setup_version || echo 0)
  [ "$CURR" != "$LAST" ] && NEED_BOOTSTRAP=1
else
  NEED_BOOTSTRAP=1
fi

if [ "$NEED_BOOTSTRAP" = 1 ]; then
  ./scripts/setup.sh
else
  source .venv/bin/activate
  pip install -U pip wheel
  pip install -r builder/requirements.txt --no-deps || true
fi