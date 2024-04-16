# C++ tests for PennyLane-Lightning

Gate implementations (kernels) are tested in `Test_GateImplementations_*.cpp` files.

As some of the kernels (AVX2 and AVX512) are only runnable on the corresponding architectures, we cannot guarantee testing all kernels on all CPU variants.
Even though it is possible to test available kernels by detecting the architecture, we currently use the approach below to simplify the test codes:

In `Test_GateImplementations_(Param|Nonparam|Matrix).cpp` files we test only the default kernels (`LM` and `PI`), of which both paradigms are architecture agnostic.

In `Test_GateImplementations_(Inverse|CompareKernels|Generator).cpp` files run tests registered to `DynamicDispatcher`. As we register kernels to `DynamicDispatcher` by detecting the runtime architecture, these files test all accessible kernels on the executing CPU architecture.
