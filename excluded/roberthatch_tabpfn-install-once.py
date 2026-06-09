# Clone and install the extensions repository, install dependencies, install tabpfn
!pip install --no-deps --target=/kaggle/working backoff filelock platformdirs posthog requests eval-type-backport pydantic-settings python-dotenv typing-inspection
!pip install tabpfn-common-utils[telemetry-interactive] --no-deps --target=/kaggle/working/
!git clone https://github.com/priorlabs/tabpfn-extensions.git
!pip install -e tabpfn-extensions --no-deps --target=/kaggle/working/
!pip install tabpfn --no-deps --target=/kaggle/working/

#   Attempting uninstall: requests
#     Found existing installation: requests 2.32.3
#     Uninstalling requests-2.32.3:
#       Successfully uninstalled requests-2.32.3
#   Attempting uninstall: platformdirs
#     Found existing installation: platformdirs 4.3.6
#     Uninstalling platformdirs-4.3.6:
#       Successfully uninstalled platformdirs-4.3.6
#   Attempting uninstall: filelock
#     Found existing installation: filelock 3.16.1
#     Uninstalling filelock-3.16.1:
#       Successfully uninstalled filelock-3.16.1
# Successfully installed backoff-2.2.1 filelock-3.20.0 platformdirs-4.5.0 posthog-6.7.9 requests-2.32.5 tabpfn-common-utils-0.2.3

#   Attempting uninstall: eval-type-backport
#     Found existing installation: eval_type_backport 0.2.0
#     Uninstalling eval_type_backport-0.2.0:
#       Successfully uninstalled eval_type_backport-0.2.0
# Successfully installed eval-type-backport-0.2.2 pydantic-settings-2.11.0 python-dotenv-1.1.1 tabpfn-2.2.1 tabpfn-extensions-0.1.6 typing-inspection-0.4.2


# !git clone https://github.com/priorlabs/tabpfn-extensions.git
# !pip install -e tabpfn-extensions


# !pip install tabpfn

