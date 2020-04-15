{% if referencefile %}
.. include:: {{ referencefile }}
{% endif %}

{{ module }}.{{ objname }}
={% for i in range(module|length) %}={% endfor %}{{ underline }}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
