import pytest
import os
from garak import _config, _plugins
import shutil

# force a local cache file to exist when this top level import is loaded
if not os.path.isfile(_plugins.PluginCache._user_plugin_cache_filename):
    _plugins.PluginCache.instance()


@pytest.fixture(autouse=True)
def config_report_cleanup(request):
    """Cleanup a testing and report directory once we are finished."""

    def remove_log_files():
        files = []
        if _config.transient.reportfile is not None:
            _config.transient.reportfile.close()
            report_html_file = _config.transient.report_filename.replace(
                ".jsonl", ".html"
            )
            hitlog_file = _config.transient.report_filename.replace(
                ".report.", ".hitlog."
            )
            files.append(_config.transient.report_filename)
            files.append(report_html_file)
            files.append(hitlog_file)

        for file in files:
            if os.path.exists(file):
                os.remove(file)

    def clear_plugin_instances():
        with _plugins.PluginProvider._mutex:
            _plugins.PluginProvider._instance_cache = {}

    request.addfinalizer(remove_log_files)
    request.addfinalizer(clear_plugin_instances)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_storage(required_space_gb=1, path='/'): Skip the test if insufficient disk space.",
    )


def check_storage(required_space_gb=1, path="/"):
    """
    Check the available disk space.

    Args:
        required_space_gb (float): Minimum required free space in GB.
        path (str): Filesystem path to check.

    Returns:
        bool: True if there is enough free space, False otherwise.
    """
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (2**30)  # Convert bytes to gigabytes

    return free_gb >= required_space_gb


def pytest_runtest_setup(item):
    """
    Called before each test is run. Performs a storage check if a specific marker is present.
    """
    marker = item.get_closest_marker("requires_storage")
    if marker:
        required_space_gb = marker.kwargs.get("required_space_gb", 1)  # Default is 1GB
        path = marker.kwargs.get("path", "/")  # Default is the root directory

        if not check_storage(required_space_gb, path):
            pytest.skip(
                f"❌ Skipping test. Not enough free space ({required_space_gb} GB) at '{path}'."
            )
        else:
            total, used, free = shutil.disk_usage(path)
            free_gb = free / (2**30)  # Convert bytes to gigabytes
            print(f"✅ Sufficient free space ({free_gb:.2f} GB) confirmed.")
