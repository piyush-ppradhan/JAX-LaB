"""
Test-suite configuration.

Most of the suite's time is XLA compilation: every test builds fresh simulation objects, and JAX compiles each of their programs
again even when an identical program was compiled before. JAX's persistent compilation cache stores compiled programs on disk,
keyed by the program itself, so identical programs are compiled once, within a run and across runs. Set JAX_LAB_TEST_CACHE to a
directory to move the cache, or to an empty string to disable it.
"""

import os

import jax

_cache_dir = os.environ.get("JAX_LAB_TEST_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "jax_lab_tests"))
if _cache_dir:
    jax.config.update("jax_compilation_cache_dir", _cache_dir)
    # Cache every program, also the small ones (the defaults skip programs that compile in under a second)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
