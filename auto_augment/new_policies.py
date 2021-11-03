"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Implements custom policies combining AutoAugment transformations
"""

from auto_augment.best_policies import ImageNetPolicy, SVHNPolicy, SubPolicy


class ImageNetPlusGeometric(ImageNetPolicy):
    def __init__(self, fillcolor=(128, 128, 128)):
        super().__init__(fillcolor=fillcolor)
        self.fill_color = fillcolor
        # add SVHN policies
        new_policies = SVHNPolicy().policies
        self.policies.extend(new_policies)
        self.add_rescaling()

    def add_rescaling(self):
        """Adds rescaling to policies"""

        rescalings = [
            SubPolicy(1.0, "rescale", 0, 1.0, "rescale", 1, self.fill_color),
            SubPolicy(1.0, "rescale", 2, 1.0, "rescale", 3, self.fill_color),
            SubPolicy(1.0, "rescale", 4, 1.0, "rescale", 5, self.fill_color),
            SubPolicy(1.0, "rescale", 6, 1.0, "rescale", 7, self.fill_color),
            SubPolicy(1.0, "rescale", 8, 1.0, "rescale", 9, self.fill_color),
        ]
        self.policies.extend(rescalings)
