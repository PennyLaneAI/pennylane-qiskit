
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

import os


DEMO_GALLERY_TEMPLATE = """
.. raw:: html

   <a href={link} data-toggle="tooltip" title="{tooltip}">
    <div class="card" style="width: 15rem; height: 19rem; margin: 10px;">
         <div class="card-header d-flex align-items-center justify-content-center" style="height: 4rem;">
              <div align="center"> <b> {name} </b> </div>
         </div>
         <div class="card-body" align="center">
           <img src="{figure}" style="max-width: 95%; max-height: 200px;" alt={name}>
         </div>
     </div>
   </a>
"""


class CustomDemoGalleryItemDirective(Directive):
    """Create a sphinx gallery style thumbnail for demos.
    Example usage:
    .. customgalleryitem::
        :name: Demo title
        :tooltip: Tooltip text
        :figure: /_static/path/to/image
        :link: https://pennylaneqiskit.readthedocs.io/en/latest/
    """

    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'name' : directives.unchanged,
                   'tooltip': directives.unchanged,
                   'figure': directives.unchanged,
                   'link': directives.unchanged}

    has_content = False
    add_index = False

    def run(self):
        try:
            if 'name' in self.options:
                name = self.options['name']
            else:
                raise ValueError('name not found')

            if 'tooltip' in self.options:
                tooltip = self.options['tooltip'][:195]
            else:
                raise ValueError('tooltip not found')

            if 'figure' in self.options:
                figure = self.options['figure']
            else:
                raise ValueError('figure not found')

            if 'link' in self.options:
                link = self.options['link']
            else:
                raise ValueError('link not found')

        except FileNotFoundError as e:
            print(e)
            return []
        except ValueError as e:
            print(e)
            raise
            return []

        thumbnail_rst = DEMO_GALLERY_TEMPLATE.format(name=name,
                                                     tooltip=tooltip,
                                                     figure=figure,
                                                     link=link)
        thumbnail = StringList(thumbnail_rst.split('\n'))
        thumb = nodes.paragraph()
        self.state.nested_parse(thumbnail, self.content_offset, thumb)
        return [thumb]
