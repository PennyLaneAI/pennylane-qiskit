# Copyright 2018-2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Custom sphinx directives
"""
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from docutils import nodes

GALLERY_TEMPLATE = """
.. raw:: html

    <div class="card" style="width: 15rem; float:left; margin: 10px;">
        <a href={link}>
            <div class="card-header">
                <b>{name}</b>
            </div>
            <div class="card-body">
                <p class="card-text"> {description} </p>
            </div>
        </a>
    </div>
"""


class CustomDeviceGalleryItemDirective(Directive):
    """Create a sphinx gallery style thumbnail.
    tooltip and figure are self explanatory. Description could be a link to
    a document like in below example.
    Example usage:

    .. customgalleryitem::
        :name: 'default.qubit'
        :description: This is a device
        :link: /path/to/device

    """

    required_arguments = 0
    optional_arguments = 4
    final_argument_whitespace = True
    option_spec = {'name': directives.unchanged,
                   'description': directives.unchanged,
                   'link': directives.unchanged}

    has_content = False
    add_index = False

    def run(self):
        try:
            if 'name' in self.options:
                name = self.options['name']

            if 'description' in self.options:
                description = self.options['description']
            else:
                raise ValueError('description not found')

            if 'link' in self.options:
                link = self.options['link']
            else:
                link = "code/qml_templates"

        except FileNotFoundError as e:
            print(e)
            return []
        except ValueError as e:
            print(e)
            raise
            return []

        thumbnail_rst = GALLERY_TEMPLATE.format(name=name,
                                                description=description,
                                                link=link)
        thumbnail = StringList(thumbnail_rst.split('\n'))
        thumb = nodes.paragraph()
        self.state.nested_parse(thumbnail, self.content_offset, thumb)
        return [thumb]
