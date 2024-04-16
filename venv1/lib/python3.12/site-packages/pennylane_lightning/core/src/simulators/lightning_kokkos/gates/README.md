# Implementation

## Expval kernels

The general multi-qubit operator kernel requires a private variable `coeffs_in` to store state vector coefficients.
In the Kokkos framework, this variable cannot be a private member of the functor, say, because all threads will overwrite it.
One must then use `TeamPolicy`s together with `scratch_memory_space` which allows creating and manipulating thread-local variables.
This implementation however appears suboptimal compared with the straightforward `RangePolicy` with bit-injection one.

The last being more verbose, it is only implemented for 1- and 2-qubit observables.
It is however possible to generate the code automatically for higher qubit counts with the following Python script. 

```python
for n_wires in range(1, 6):
    name = f"getExpVal{n_wires}QubitOpFunctor"
    nU = 2**n_wires

    print(
        f"""template <class PrecisionT, std::size_t n_wires> struct {name} {{
        
        using ComplexT = Kokkos::complex<PrecisionT>;
        using KokkosComplexVector = Kokkos::View<ComplexT *>;
        using KokkosIntVector = Kokkos::View<std::size_t *>;

        KokkosComplexVector arr;
        KokkosComplexVector matrix;
        KokkosIntVector wires;
        std::size_t dim;
        std::size_t num_qubits;

        {name}(const KokkosComplexVector &arr_,
        const std::size_t num_qubits_,
                                    const KokkosComplexVector &matrix_,
                                    const KokkosIntVector &wires_) {{
            wires = wires_;
            arr = arr_;
            matrix = matrix_;
            num_qubits = num_qubits_;
            dim = static_cast<std::size_t>(1U) << wires.size();
        }}
        
        KOKKOS_INLINE_FUNCTION
        void operator()(const std::size_t k, PrecisionT &expval) const {{
        const std::size_t kdim = k * dim;
    """
    )

    for k in range(nU):
        print(
            f"""
        std::size_t i{k:0{n_wires}b} = kdim | {k};
        for (std::size_t pos = 0; pos < n_wires; pos++) {{
            std::size_t x =
                ((i{k:0{n_wires}b} >> (n_wires - pos - 1)) ^
                    (i{k:0{n_wires}b} >> (num_qubits - wires(pos) - 1))) &
                1U;
            i{k:0{n_wires}b} = i{k:0{n_wires}b} ^ ((x << (n_wires - pos - 1)) |
                            (x << (num_qubits - wires(pos) - 1)));
        }}
        """
        )

    # print("expval += real(")
    for k in range(nU):
        tmp = f"expval += real(conj(arr(i{k:0{n_wires}b})) * ("
        tmp += f"matrix(0B{k:0{n_wires}b}{0:0{n_wires}b}) * arr(i{0:0{n_wires}b})"
        for j in range(1, nU):
            tmp += (
                f" + matrix(0B{k:0{n_wires}b}{j:0{n_wires}b}) * arr(i{j:0{n_wires}b})"
            )
        print(tmp, end="")
        print("));")
    print("}")
    print("};")
```

## Gate kernels

Same goes for multi-qubit gates.
It is only implemented for 1- and 2-qubit observables.
It is however possible to generate the code automatically for higher qubit counts with the following Python script. 

```python
for n_wires in range(1, 6):
    name = f"apply{n_wires}QubitOpFunctor"
    nU = 2**n_wires

    print(f"""template <class PrecisionT, std::size_t n_wires, bool inverse=false> struct {name} {{
        
        using ComplexT = Kokkos::complex<PrecisionT>;
        using KokkosComplexVector = Kokkos::View<ComplexT *>;
        using KokkosIntVector = Kokkos::View<std::size_t *>;

        KokkosComplexVector arr;
        KokkosComplexVector matrix;
        KokkosIntVector wires;
        std::size_t dim;
        std::size_t num_qubits;

        {name}(KokkosComplexVector &arr_,
                                    std::size_t num_qubits_,
                                    const KokkosComplexVector &matrix_,
                                    KokkosIntVector &wires_) {{
            wires = wires_;
            arr = arr_;
            matrix = matrix_;
            num_qubits = num_qubits_;
            dim = static_cast<std::size_t>(1U) << n_wires;
        }}
        
        KOKKOS_INLINE_FUNCTION
        void operator()(const std::size_t k) const {{
        const std::size_t kdim = k * dim;
    """)

    for k in range(nU):
        print(f"""    std::size_t i{k:0{n_wires}b} = kdim | {k};
        for (std::size_t pos = 0; pos < n_wires; pos++) {{
            std::size_t x =
                ((i{k:0{n_wires}b} >> (n_wires - pos - 1)) ^
                    (i{k:0{n_wires}b} >> (num_qubits - wires(pos) - 1))) &
                1U;
            i{k:0{n_wires}b} = i{k:0{n_wires}b} ^ ((x << (n_wires - pos - 1)) |
                            (x << (num_qubits - wires(pos) - 1)));
        }}
        ComplexT v{k:0{n_wires}b} = arr(i{k:0{n_wires}b});
        """)

    print("if constexpr(inverse){")
    for k in range(nU):
        tmp = f"    arr(i{k:0{n_wires}b}) = "
        tmp += f"conj(matrix(0B{0:0{n_wires}b}{k:0{n_wires}b})) * v{0:0{n_wires}b}"
        for j in range(1, nU):
            tmp += f" + conj(matrix(0B{j:0{n_wires}b}{k:0{n_wires}b})) * v{j:0{n_wires}b}"
        print(tmp, end="")
        print(";")
    print("}", end="")
    print("else{")
    for k in range(nU):
        tmp = f"    arr(i{k:0{n_wires}b}) = "
        tmp += f"matrix(0B{k:0{n_wires}b}{0:0{n_wires}b}) * v{0:0{n_wires}b}"
        for j in range(1, nU):
            tmp += f" + matrix(0B{k:0{n_wires}b}{j:0{n_wires}b}) * v{j:0{n_wires}b}"
        print(tmp, end="")
        print(";")
    print("}")
    print("}")
    print("};")
```