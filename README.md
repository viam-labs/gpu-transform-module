# gpu-transform-module

Module to redirect transforms from cameras on the vision service to use GPU CUDA instead of CPU to improve processing speed and efficiency.

<pre>
├── gpu_transform_module.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   └── top_level.txt
├── image_object.py
├── meta.json
├── README.md
├── requirements.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── camera_module.py
│   ├── config.py
│   ├── transform_pipeline.py
│   └── utils
│       ├── __init__.py
│       └── gpu_utils.py
└── tests
    └── test_transform_pipeline.py
</pre>

Run tests through pytest with:
<pre> pytest tests/ -v </pre>