{% block structs %}{% endblock structs %}

{% for binding in bindings %}
[[group(0), binding({{ binding.counter }})]]
var<storage, read_write> {{ binding.tensor }}: {{ binding.inner_type }};
{% endfor %}

{% block main %}{% endblock main %}