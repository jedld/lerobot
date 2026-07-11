#!/usr/bin/env bash
# Shared ROCm library path helpers for Pi SO-101 scripts.
# Prefer the molmoact2 helpers when available so both stacks stay in sync.

resolve_rocm_lib_dirs() {
  local -a dirs=()
  local d seen="|"

  _add_dir() {
    local candidate="$1"
    [[ -d "$candidate" ]] || return 0
    case "$seen" in
      *"|${candidate}|"*) return 0 ;;
    esac
    dirs+=("$candidate")
    seen="${seen}${candidate}|"
  }

  _add_dir /opt/rocm/lib
  _add_dir /opt/rocm/lib64
  for d in /opt/rocm-*/lib /opt/rocm-*/lib64; do
    _add_dir "$d"
  done

  if ((${#dirs[@]} == 0)); then
    return 1
  fi
  printf '%s\n' "${dirs[@]}"
}

export_rocm_ld_library_path() {
  local dir joined=""
  while IFS= read -r dir; do
    [[ -n "$dir" ]] || continue
    if [[ -n "$joined" ]]; then
      joined="${joined}:"
    fi
    joined="${joined}${dir}"
  done < <(resolve_rocm_lib_dirs 2>/dev/null || true)

  if [[ -n "$joined" ]]; then
    export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
}

is_amd_strix() {
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi 2>/dev/null | grep -qiE 'Radeon 880M|Radeon 890M' && return 0
  fi
  if command -v lspci >/dev/null 2>&1; then
    lspci 2>/dev/null | grep -qiE 'Radeon 880M|Radeon 890M|Strix' && return 0
  fi
  return 1
}
