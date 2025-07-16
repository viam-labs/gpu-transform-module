# Module gpu-transform-camera 

Provide a description of the purpose of the module and any relevant information.

## Model isha-org:gpu-transform-camera:gpu-transform

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
├── setup.sh
|-- build.sh
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

### Configuration
The following attribute template can be used to configure this model:

```json
{
"attribute_1": <float>,
"attribute_2": <string>
}
```

#### Attributes

The following attributes are available for this model:

| Name          | Type   | Inclusion | Description                |
|---------------|--------|-----------|----------------------------|
| `attribute_1` | float  | Required  | Description of attribute 1 |
| `attribute_2` | string | Optional  | Description of attribute 2 |

#### Example Configuration

```json
{
  "attribute_1": 1.0,
  "attribute_2": "foo"
}
```

### DoCommand

If your model implements DoCommand, provide an example payload of each command that is supported and the arguments that can be used. If your model does not implement DoCommand, remove this section.

#### Example DoCommand

```json
{
  "command_name": {
    "arg1": "foo",
    "arg2": 1
  }
}
```
