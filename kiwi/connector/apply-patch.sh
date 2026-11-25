#!/usr/bin/env bash
# Patch the connector installed here for OpenAI tool calls. install.sh runs it; it is separate to re-run after
# reinstalling the connector by hand, or after the patch changed. Does nothing if the patch is already in.
#
# Without it, the connector (2.2, as 2.0.0) returns tool calls with no "id" and no "type", drops the
# "tool_call_id" of a tool's result, and gives no "finish_reason": an agent framework cannot tie a result back to its
# call, drops the call, and the model asks for it again forever -- strands-agents logs "incomplete tool use block,
# skipping". That much is the iotc-eiq-imx95-property-alarm demo's patch, models.py and Llm.py.
# HfLlm.py is ours: the model's prompt shows an earlier call's arguments quoted and escaped, the model writes its next
# call the same way, and the connector cannot read it back -- a second call with arguments, after a first with
# arguments, comes back as text. So is Llm.py's repetition penalty: the config's and a request's repeat_penalty passed
# on to the Ara240, which takes it, where 2.2 reads it and drops it.
set -e
cd "$(dirname "$(readlink -f "$0")")"

PACKAGE=$(echo venv/lib/python3.*/site-packages/eiq_aaf_connector)
[ -d "$PACKAGE" ] || { echo "The connector is not installed here: connector/install.sh first."; exit 1; }

if grep -q "repetition_penalty = llm_params.repeat_penalty" "$PACKAGE/llm_engines/Llm.py"; then
  echo "Already patched: $PACKAGE"
  exit 0
fi
if grep -q "uuid4().hex" "$PACKAGE/models/models.py"; then
  echo "An older patch is in: putting the connector's own files back first..."
  venv/bin/pip install --disable-pip-version-check --quiet --force-reinstall --no-deps \
    /usr/share/python-wheels/eiq_aaf_connector-*.whl
fi

# The files are in two directories: each part of the patch to its own.
sed -n '/^--- a\/models.py/,/^--- a\/Llm.py/p' connector-openai-fix.patch | sed '$d' \
  | patch -d "$PACKAGE/models" -p1 -F3 --quiet --no-backup-if-mismatch
sed -n '/^--- a\/Llm.py/,$p' connector-openai-fix.patch \
  | patch -d "$PACKAGE/llm_engines" -p1 -F3 --quiet --no-backup-if-mismatch
echo "Patched: $PACKAGE"
