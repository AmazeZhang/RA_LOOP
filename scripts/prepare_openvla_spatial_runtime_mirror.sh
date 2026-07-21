#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=/home/imc/models/ra-loop/openvla-oft-spatial
RUNTIME_ROOT=/home/imc/yzy/RA_LOOP/runtime/openvla-oft-spatial-smoke

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "Missing source checkpoint: ${SOURCE_ROOT}" >&2
  exit 1
fi

if [[ -e "${RUNTIME_ROOT}" ]]; then
  echo "Refusing to overwrite existing runtime mirror: ${RUNTIME_ROOT}" >&2
  exit 2
fi

mkdir -p "${RUNTIME_ROOT}"
shopt -s dotglob nullglob

for source_entry in "${SOURCE_ROOT}"/*; do
  entry_name="$(basename "${source_entry}")"
  case "${entry_name}" in
    .cache|config.json.back.*)
      continue
      ;;
    config.json|modeling_prismatic.py|configuration_prismatic.py)
      cp -p "${source_entry}" "${RUNTIME_ROOT}/${entry_name}"
      ;;
    *)
      ln -s "${source_entry}" "${RUNTIME_ROOT}/${entry_name}"
      ;;
  esac
done

echo "Runtime mirror created: ${RUNTIME_ROOT}"
echo "Copied mutable metadata: config.json, modeling_prismatic.py, configuration_prismatic.py"
echo "All model weights remain read-only references to: ${SOURCE_ROOT}"
