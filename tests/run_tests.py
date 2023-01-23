from tests.models_tests.discriminator_tests.disc_patch1_tests import DiscPatch1Tests
from tests.models_tests.discriminator_tests.disc_patch16_tests import DiscPatch16Tests
from tests.models_tests.discriminator_tests.disc_patch70_tests import DiscPatch70Tests
from tests.models_tests.discriminator_tests.disc_patch286_tests import DiscPatch256Tests

from tests.models_tests.generators_tests.unet_generator_tests import UNETGenTests
from tests.models_tests.generators_tests.encoder_decoder_generator_tests import EncDecGenTests


if __name__ == '__main__':

    print('Discriminator tests')
    print('   Patch 1 discriminator tests')
    DiscPatch1Tests()
    print('   Patch 256 discriminator tests')
    DiscPatch256Tests()
    print('   Patch 70 discriminator tests')
    DiscPatch70Tests()
    print('   Patch 16 discriminator tests')
    DiscPatch16Tests()

    print('Generators tests')
    print('   UNET generator tests')
    UNETGenTests()
    print('   Encoder-Decoder generator tests')
    EncDecGenTests()
