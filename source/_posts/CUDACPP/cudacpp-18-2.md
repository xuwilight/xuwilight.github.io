---
title: CUDA C++ 笔记（十二）第17章——C++ Language Support（二）
date: 2024-07-23 20:00:00
tags: [CUDA, CUDA C++]
categories: [CUDA C++ Programming Guide]
description: 第17章 C++ Language Support 第二部分，介绍多态函数包装器 nvstd::function、扩展 Lambda（Extended Lambdas）的使用与限制，以及设备代码中的 C++ 代码示例。
---

# 第18章 C++ Language Support Part2

**Page 428~450**

17.6-17.8

**主要内容：**

1. Polymorphic Function Wrappers
2. Extended Lambdas
3. Code Samples

## 17.6 Polymorphic Function Wrappers

这段内容详细介绍了 CUDA C++ 中的**多态函数包装器** `nvstd::function`，它是定义在 `nvfunctional` 头文件中的一个类模板。

`nvstd::function` 的主要作用是存储、复制和调用任何可调用目标（如函数、lambda 表达式等）。它可以在主机代码（host code）和设备代码（device code）中使用。

```cpp
#include <nvfunctional>

// 只能在设备代码中调用
__device__ int foo_d() { return 1; }

// 既可以在主机代码中调用，也可以在设备代码中调用
__host__ __device__ int foo_hd() { return 2; }

// 只能在主机代码中调用
__host__ int foo_h() { return 3; }

//
__global__ void kernel(int *result) {
    // 使用 nvstd::function 存储设备函数 foo_d
    nvstd::function<int()> fn1 = foo_d;

    // 使用 nvstd::function 存储主机-设备函数 foo_hd
    nvstd::function<int()> fn2 = foo_hd;

    // 使用 nvstd::function 存储一个返回 10 的 lambda 表达式
    nvstd::function<int()> fn3 = []() { return 10; };

    // 调用 fn1、fn2、fn3，并将结果相加，存储到 result 指向的内存中
    *result = fn1() + fn2() + fn3();
}

//
__host__ __device__ void hostdevice_func(int *result) {
    // 使用 nvstd::function 存储主机-设备函数 foo_hd
    nvstd::function<int()> fn1 = foo_hd;

    // 使用 nvstd::function 存储一个返回 10 的 lambda 表达式
    nvstd::function<int()> fn2 = []() { return 10; };

    // 调用 fn1、fn2，并将结果相加，存储到 result 指向的内存中
    *result = fn1() + fn2();
}

//
__host__ void host_func(int *result) {
    // 使用 nvstd::function 存储主机函数 foo_h
    nvstd::function<int()> fn1 = foo_h;

    // 使用 nvstd::function 存储主机-设备函数 foo_hd
    nvstd::function<int()> fn2 = foo_hd;

    // 使用 nvstd::function 存储一个返回 10 的 lambda 表达式
    nvstd::function<int()> fn3 = []() { return 10; };

    // 调用 fn1、fn2、fn3，并将结果相加，存储到 result 指向的内存中
    *result = fn1() + fn2() + fn3();
}
```

在主机代码中，`nvstd::function` 的实例**不能**用 `__device__` 函数的地址初始化，也不能用 `operator()` 是 `__device__` 函数的函数对象（functor）初始化。

在设备代码中，`nvstd::function` 的实例**不能**用 `__host__` 函数的地址初始化，也不能用 `operator()` 是 `__host__` 函数的函数对象初始化。

`nvstd::function` 的实例**不能**在运行时从主机代码传递到设备代码（反之亦然）。如果 `__global__` 函数是从主机代码启动的，则 `nvstd::function` **不能**用作 `__global__` 函数的参数类型。

1. 主机代码中的限制：

在主机代码中，`nvstd::function` 只能存储主机函数或主机-设备函数（`__host__` 或 `__host__ __device__`）。

如果尝试用 `__device__` 函数的地址初始化 `nvstd::function`，会导致编译错误。

同样，如果尝试用一个 `operator()` 是 `__device__` 函数的函数对象初始化 `nvstd::function`，也会导致编译错误。

原因：主机代码无法直接调用设备函数，因为设备函数只能在设备上执行。

2. 设备代码中的限制：

在设备代码中，`nvstd::function` 只能存储设备函数或主机-设备函数（`__device__` 或 `__host__ __device__`）。

如果尝试用 `__host__` 函数的地址初始化 `nvstd::function`，会导致编译错误。

同样，如果尝试用一个 `operator()` 是 `__host__` 函数的函数对象初始化 `nvstd::function`，也会导致编译错误。

原因：设备代码无法直接调用主机函数，因为主机函数只能在主机上执行。

3. 跨主机和设备传递的限制：

`nvstd::function` 的实例不能在运行时从主机代码传递到设备代码，反之亦然。

例如，不能将主机代码中创建的 `nvstd::function` 实例作为参数传递给设备代码中的函数，也不能将设备代码中创建的 `nvstd::function` 实例传递回主机代码。

原因：`nvstd::function` 的实例可能包含指向主机或设备内存的指针，而这些指针在另一种上下文中是无效的。

4. `__global__` 函数参数的限制：

如果 `__global__` 函数是从主机代码启动的，则 `nvstd::function` 不能用作 `__global__` 函数的参数类型。

原因：`nvstd::function` 的实例可能包含主机代码的上下文信息，无法直接传递给设备代码。

### 一句话总结

`nvstd::function` 在主机和设备代码中有严格的函数类型限制，且不能跨主机和设备传递，也不能用于 `__global__` 函数的参数。

```cpp
#include <nvfunctional>  // 包含 nvstd::function 的头文件


__device__ int foo_d() { return 1; }


__host__ int foo_h() { return 3; }


auto lam_h = [] { return 0; };


__global__ void k(void) {
    // 错误：尝试用主机函数 foo_h 的地址初始化 nvstd::function
    // 原因：设备代码中不能使用主机函数
    nvstd::function<int()> fn1 = foo_h;

    // 错误：尝试用主机 lambda 表达式 lam_h 初始化 nvstd::function
    // 原因：设备代码中不能使用主机 lambda 表达式（其 operator() 是主机函数）
    nvstd::function<int()> fn2 = lam_h;
}


__global__ void kern(nvstd::function<int()> f1) { }

// 主机函数
void foo(void) {
    // 错误：尝试用设备函数 foo_d 的地址初始化 nvstd::function
    // 原因：主机代码中不能使用设备函数
    nvstd::function<int()> fn1 = foo_d;

    // 设备 lambda 表达式：只能在设备代码中调用
    auto lam_d = [=] __device__ { return 1; };

    // 错误：尝试用设备 lambda 表达式 lam_d 初始化 nvstd::function
    // 原因：主机代码中不能使用设备 lambda 表达式（其 operator() 是设备函数）
    nvstd::function<int()> fn2 = lam_d;

    // 错误：尝试将 nvstd::function 从主机代码传递到设备代码
    // 原因：nvstd::function 的实例不能在主机和设备之间传递
    kern<<<1,1>>>(fn2);
}
```

**`nvstd::function` 在 `nvfunctional` 头文件中定义如下**

```cpp
namespace nvstd {
    // 定义类模板 function，用于包装可调用对象（如函数、lambda 表达式等）
    // _RetType 是函数的返回类型，_ArgTypes... 是函数的参数类型（可变模板参数）
    template <class _RetType, class ..._ArgTypes>
    class function<_RetType(_ArgTypes...)> {
    public:
        // 构造函数：默认构造函数，创建一个空的 function 对象
        __device__ __host__ function() noexcept;

        // 构造函数：从 nullptr 初始化，创建一个空的 function 对象
        __device__ __host__ function(nullptr_t) noexcept;

        // 拷贝构造函数：从另一个 function 对象拷贝构造
        __device__ __host__ function(const function &);

        // 移动构造函数：从另一个 function 对象移动构造
        __device__ __host__ function(function &&);

        // 模板构造函数：从任意可调用对象 _F（如函数指针、lambda 表达式等）构造 function 对象
        template<class _F>
        __device__ __host__ function(_F);

        // 析构函数：释放 function 对象占用的资源
        __device__ __host__ ~function();

        // 赋值运算符：从另一个 function 对象拷贝赋值
        __device__ __host__ function& operator=(const function&);

        // 赋值运算符：从另一个 function 对象移动赋值
        __device__ __host__ function& operator=(function&&);

        // 赋值运算符：将 function 对象设置为空（等同于 nullptr）
        __device__ __host__ function& operator=(nullptr_t);

        // 模板赋值运算符：从任意可调用对象 _F 赋值
        __device__ __host__ function& operator=(_F&&);

        // 交换函数：交换两个 function 对象的内容
        __device__ __host__ void swap(function&) noexcept;

        // 函数容量检查：检查 function 对象是否包含一个有效的可调用对象
        // 返回 true 表示有可调用对象，false 表示为空
        __device__ __host__ explicit operator bool() const noexcept;

        // 函数调用运算符：调用 function 对象中存储的可调用对象
        // 参数类型为 _ArgTypes...，返回类型为 _RetType
        __device__ _RetType operator()(_ArgTypes...) const;
    };

    // 空指针比较运算符：判断 function 对象是否等于 nullptr
    template <class _R, class... _ArgTypes>
    __device__ __host__ bool operator==(const function<_R(_ArgTypes...)>&, nullptr_t) noexcept;

    // 空指针比较运算符：判断 nullptr 是否等于 function 对象
    template <class _R, class... _ArgTypes>
    __device__ __host__ bool operator==(nullptr_t, const function<_R(_ArgTypes...)>&) noexcept;

    // 空指针比较运算符：判断 function 对象是否不等于 nullptr
    template <class _R, class... _ArgTypes>
    __device__ __host__ bool operator!=(const function<_R(_ArgTypes...)>&, nullptr_t) noexcept;

    // 空指针比较运算符：判断 nullptr 是否不等于 function 对象
    template <class _R, class... _ArgTypes>
    __device__ __host__ bool operator!=(nullptr_t, const function<_R(_ArgTypes...)>&) noexcept;

    // 特化的交换函数：交换两个 function 对象的内容
    template <class _R, class... _ArgTypes>
    __device__ __host__ void swap(function<_R(_ArgTypes...)>&, function<_R(_ArgTypes...)>&);

} 
```

## 17.7 Extended Lambdas

nvcc 编译器的 `--extended-lambda` 标志允许在 lambda 表达式中显式指定执行空间注解（execution space annotations）。

这些注解应位于 lambda-introducer 之后，可选的 lambda-declarator 之前。

当指定了 `--extended-lambda` 标志时，nvcc 会定义宏 `__CUDACC_EXTENDED_LAMBDA__`。

**说人话：**

1. `--extended-lambda` 的作用：

它让你可以在 lambda 表达式中明确指定这个 lambda 是在设备上运行（`__device__`），还是同时在主机和设备上运行（`__host__ __device__`）。

2. 注解的位置：

这些注解（如 `__device__`）需要写在 lambda 的 `[]` 后面，参数列表和返回类型的前面。例如：

```cpp
auto lam = [] __device__ { return 42; }; // 这是一个设备 lambda
```

3. 宏 `__CUDACC_EXTENDED_LAMBDA__`：

当你用了 `--extended-lambda` 标志，nvcc 会自动定义这个宏，方便你在代码中检查是否启用了扩展 lambda 功能。

扩展的 `__device__` lambda：显式使用 `__device__` 注解的 lambda 表达式，并且定义在 `__host__` 或 `__host__ __device__` 函数的直接或嵌套块作用域内。

扩展的 `__host__ __device__` lambda：显式使用 `__host__` 和 `__device__` 注解的 lambda 表达式，并且定义在 `__host__` 或 `__host__ __device__` 函数的直接或嵌套块作用域内。

扩展 lambda：指扩展的 `__device__` lambda 或扩展的 `__host__ __device__` lambda。扩展 lambda 可以用于 `__global__` 函数模板实例化的类型参数中。

如果未显式指定执行空间注解，则根据与 lambda 关联的闭包类的作用域计算执行空间注解，如 C++11 支持部分所述。执行空间注解会应用于与 lambda 关联的闭包类的所有方法。

```cpp
// 示例：展示扩展 lambda 的使用
void foo_host(void) {
    // 不是扩展 lambda：没有显式指定执行空间注解
    auto lam1 = [] { };

    // 扩展的 __device__ lambda
    auto lam2 = [] __device__ { };

    // 扩展的 __host__ __device__ lambda
    auto lam3 = [] __host__ __device__ { };

    // 不是扩展 lambda：显式指定了 __host__，但没有 __device__
    auto lam4 = [] __host__ { };
}

// __host__ __device__ 函数
__host__ __device__ void foo_host_device(void) {
    // 不是扩展 lambda：没有显式指定执行空间注解
    auto lam1 = [] { };

    // 扩展的 __device__ lambda
    auto lam2 = [] __device__ { };

    // 扩展的 __host__ __device__ lambda
    auto lam3 = [] __host__ __device__ { };

    // 不是扩展 lambda：显式指定了 __host__，但没有 __device__
    auto lam4 = [] __host__ { };
}

// __device__ 函数
__device__ void foo_device(void) {
    // 此函数中的 lambda 都不是扩展 lambda，
    // 因为外层函数不是 __host__ 或 __host__ __device__ 函数。
    auto lam1 = [] { };
    auto lam2 = [] __device__ { };
    auto lam3 = [] __host__ __device__ { };
    auto lam4 = [] __host__ { };
}

// lam1 和 lam2 不是扩展 lambda，因为它们没有定义在 __host__ 或 __host__ __device__ 函数中。
auto lam1 = [] { };
auto lam2 = [] __host__ __device__ { };
```

### 扩展 Lambda 的类型特性

编译器提供了类型特性（type traits），用于在编译时检测扩展 lambda 的闭包类型：

- `__nv_is_extended_device_lambda_closure_type(type)`：如果 type 是扩展的 `__device__` lambda 的闭包类，则返回 `true`。
- `__nv_is_extended_device_lambda_with_preserved_return_type(type)`：如果 type 是扩展的 `__device__` lambda 的闭包类，并且 lambda 使用尾随返回类型定义，则返回 `true`。
- `__nv_is_extended_host_device_lambda_closure_type(type)`：如果 type 是扩展的 `__host__ __device__` lambda 的闭包类，则返回 `true`。

这些特性可以在所有编译模式下使用，无论是否启用了 lambda 或扩展 lambda。

```cpp
// 定义宏，用于检测扩展 lambda 的闭包类型
#define IS_D_LAMBDA(X) __nv_is_extended_device_lambda_closure_type(X) // 检测是否为扩展的 __device__ lambda
#define IS_DPRT_LAMBDA(X) __nv_is_extended_device_lambda_with_preserved_return_type(X) // 检测是否为扩展的 __device__ lambda 并保留返回类型
#define IS_HD_LAMBDA(X) __nv_is_extended_host_device_lambda_closure_type(X) // 检测是否为扩展的 __host__ __device__ lambda

// 定义一个全局的 __host__ __device__ lambda
auto lam0 = [] __host__ __device__ { };

// 定义一个函数 foo
void foo(void) {
    // 定义一个普通的 lambda，没有执行空间注解
    auto lam1 = [] { };

    // 定义一个扩展的 __device__ lambda
    auto lam2 = [] __device__ { };

    // 定义一个扩展的 __host__ __device__ lambda
    auto lam3 = [] __host__ __device__ { };

    // 定义一个扩展的 __device__ lambda，并使用尾随返回类型（返回 double）
    auto lam4 = [] __device__ () -> double { return 3.14; };

    // 定义一个扩展的 __device__ lambda，并使用尾随返回类型（返回 decltype(&x)）
    auto lam5 = [] __device__ (int x) -> decltype(&x) { return 0; };

    // lam0 不是扩展 lambda，因为它定义在函数作用域外
    static_assert(!IS_D_LAMBDA(decltype(lam0)), "lam0 should not be an extended __device__ lambda");
    static_assert(!IS_DPRT_LAMBDA(decltype(lam0)), "lam0 should not be an extended __device__ lambda with preserved return type");
    static_assert(!IS_HD_LAMBDA(decltype(lam0)), "lam0 should not be an extended __host__ __device__ lambda");

    // lam1 不是扩展 lambda，因为它没有执行空间注解
    static_assert(!IS_D_LAMBDA(decltype(lam1)), "lam1 should not be an extended __device__ lambda");
    static_assert(!IS_DPRT_LAMBDA(decltype(lam1)), "lam1 should not be an extended __device__ lambda with preserved return type");
    static_assert(!IS_HD_LAMBDA(decltype(lam1)), "lam1 should not be an extended __host__ __device__ lambda");

    // lam2 是扩展的 __device__ lambda
    static_assert(IS_D_LAMBDA(decltype(lam2)), "lam2 should be an extended __device__ lambda");
    static_assert(!IS_DPRT_LAMBDA(decltype(lam2)), "lam2 should not be an extended __device__ lambda with preserved return type");
    static_assert(!IS_HD_LAMBDA(decltype(lam2)), "lam2 should not be an extended __host__ __device__ lambda");

    // lam3 是扩展的 __host__ __device__ lambda
    static_assert(!IS_D_LAMBDA(decltype(lam3)), "lam3 should not be an extended __device__ lambda");
    static_assert(!IS_DPRT_LAMBDA(decltype(lam3)), "lam3 should not be an extended __device__ lambda with preserved return type");
    static_assert(IS_HD_LAMBDA(decltype(lam3)), "lam3 should be an extended __host__ __device__ lambda");

    // lam4 是扩展的 __device__ lambda，并且保留了返回类型
    static_assert(IS_D_LAMBDA(decltype(lam4)), "lam4 should be an extended __device__ lambda");
    static_assert(IS_DPRT_LAMBDA(decltype(lam4)), "lam4 should be an extended __device__ lambda with preserved return type");
    static_assert(!IS_HD_LAMBDA(decltype(lam4)), "lam4 should not be an extended __host__ __device__ lambda");

    // lam5 是扩展的 __device__ lambda，但没有保留返回类型，因为尾随返回类型引用了 lambda 参数名称
    static_assert(IS_D_LAMBDA(decltype(lam5)), "lam5 should be an extended __device__ lambda");
    static_assert(!IS_DPRT_LAMBDA(decltype(lam5)), "lam5 should not be an extended __device__ lambda with preserved return type");
    static_assert(!IS_HD_LAMBDA(decltype(lam5)), "lam5 should not be an extended __host__ __device__ lambda");
}
```

在调用主机编译器之前，CUDA 编译器会将扩展 lambda 表达式替换为命名空间作用域中定义的占位符类型（placeholder type）的实例。

占位符类型的模板参数需要获取包含原始扩展 lambda 表达式的函数的地址。这对于任何模板参数涉及扩展 lambda 闭包类型的 `__global__` 函数模板的正确执行是必需的。

根据定义，扩展 lambda 存在于 `__host__` 或 `__host__ __device__` 函数的直接或嵌套块作用域内。

如果该函数不是 lambda 表达式的 `operator()`，则它被视为扩展 lambda 的包含函数。

否则，扩展 lambda 定义在一个或多个外层 lambda 表达式的 `operator()` 的直接或嵌套块作用域内。如果最外层的此类 lambda 表达式定义在函数 F 的直接或嵌套块作用域内，则 F 是计算得到的包含函数；否则，包含函数不存在。

```cpp
// 定义一个函数 foo
void foo(void) {
    // lam1 是一个扩展的 __device__ lambda
    // 它的包含函数是 foo，因为它直接定义在 foo 的作用域内
    auto lam1 = [] __device__ { };

    // lam2 是一个普通的 lambda，没有执行空间注解
    auto lam2 = [] {
        // lam3 是一个嵌套的普通 lambda
        auto lam3 = [] {
            // lam4 是一个扩展的 __host__ __device__ lambda
            // 它的包含函数是 foo，因为它位于 foo 的嵌套作用域内
            auto lam4 = [] __host__ __device__ { };
        };
    };
}

// lam6 是一个全局的普通 lambda
auto lam6 = [] {
    // lam7 是一个扩展的 __host__ __device__ lambda
    // 它的包含函数不存在，因为它定义在全局作用域中，而不是任何 __host__ 或 __host__ __device__ 函数的作用域内
    auto lam7 = [] __host__ __device__ { };
};
```

总结来说，CUDA 编译器通过确定扩展 lambda 的"包含函数"来确保其正确执行，而"包含函数"的确定依赖于 lambda 所在的上下文和作用域。

以下是扩展 lambda 的限制：

#### 1. 扩展 Lambda 不能嵌套在另一个扩展 Lambda 中

```cpp
void foo(void) {
    auto lam1 = [] __host__ __device__ {
        // 错误：扩展 lambda 不能嵌套在另一个扩展 lambda 中
        auto lam2 = [] __host__ __device__ { };
    };
}
```

#### 2. 扩展 Lambda 不能定义在泛型 Lambda 中

```cpp
void foo(void) {
    auto lam1 = [] (auto) {
        // 错误：扩展 lambda 不能定义在泛型 lambda 中
        auto lam2 = [] __host__ __device__ { };
    };
}
```

#### 3. 扩展 Lambda 必须定义在函数的直接或嵌套作用域内

```cpp
auto lam1 = [] {
    // 错误：最外层的 lambda 没有定义在函数的作用域内
    auto lam2 = [] __host__ __device__ { };
};
```

#### 4. 扩展 Lambda 的包含函数必须具有可获取的地址

```cpp
void foo(void) {
    // 合法：扩展 lambda 定义在函数作用域内
    auto lam1 = [] __device__ { return 0; };
}

struct S1_t {
    S1_t(void) {
        // 错误：无法获取构造函数地址
        auto lam4 = [] __device__ { return 0; };
    }
};

class C0_t {
    void foo(void) {
        // 错误：成员函数具有私有访问权限
        auto temp1 = [] __device__ { return 10; };
    }
};
```

#### 5. 扩展 Lambda 的包含函数必须具有明确的地址

```cpp
template <typename Bar>
void A<Bar>::test() {
    // 错误：类 typedef 'Bar' 遮蔽了模板参数 'Bar'
    auto lam1 = [] __host__ __device__ { return 4; };
}
```

#### 6. 扩展 Lambda 不能定义在函数的局部类中

```cpp
void foo(void) {
    struct S1_t {
        void bar(void) {
            // 错误：扩展 lambda 不能定义在函数的局部类中
            auto lam4 = [] __host__ __device__ { return 0; };
        }
    };
}
```

#### 7. 扩展 Lambda 的包含函数不能有推导的返回类型

```cpp
auto foo(void) {
    // 错误：扩展 lambda 的包含函数不能有推导的返回类型
    auto lam1 = [] __host__ __device__ { return 0; };
}
```

#### 8. `__host__ __device__` 扩展 Lambda 不能是泛型 Lambda

```cpp
void foo(void) {
    // 错误：`__host__ __device__` 扩展 lambda 不能是泛型 lambda
    auto lam1 = [] __host__ __device__ (auto i) { return i; };
}
```

#### 9. 模板函数的限制

```cpp
template <typename T>
void bar4(void) {
    auto lam1 = [] __device__ { return 10; };
}

int main() {
    struct S1_t { };
    // 错误：模板实例化参数类型不能是函数的局部类型
    bar4<S1_t>();
}
```

#### 10. Visual Studio 编译器的限制

在 Visual Studio 编译器中，扩展 lambda 的包含函数必须具有外部链接。

扩展 lambda 不能定义在 `if-constexpr` 块中。

#### 11. 捕获变量的限制

```cpp
void foo(void) {
    int a = 1;
    // 错误：扩展 lambda 不能按引用捕获变量
    auto lam3 = [&a] __device__ () { return a; };
}
```

#### 12. `if-constexpr` 块中的捕获限制

```cpp
void foo(void) {
    int yyy = 4;
    auto lam9 = [=] __device__ {
        int result = 0;
        if constexpr(false) {
            // 错误：`yyy` 不能在 `if-constexpr` 块中首次捕获
            result += yyy;
        }
        return result;
    };
}
```

#### 13. 扩展 Lambda 的顺序依赖性

```cpp
__host__ __device__ void foo(void) {
#if defined(__CUDA_ARCH__)
    auto lam1 = [] __device__ { return 0; };
#endif
    auto lam2 = [] __device__ { return 4; };
}
```

#### 14. 扩展 Lambda 的返回类型推断限制

```cpp
void foo(void) {
    auto lam1 = [] __device__ { return "10"; };
    // 错误：不能在主机代码中推断扩展 lambda 的返回类型
    std::result_of<decltype(lam1)()>::type xx1 = "abc";
}
```

#### 15. 扩展 Lambda 的参数类型推断限制

扩展 lambda 的参数类型只能在设备代码中推断。

#### 16. 扩展 Lambda 的闭包类布局

```cpp
void foo(void) {
    int x1 = 1;
    auto lam1 = [=] __host__ __device__ {
#ifdef __CUDA_ARCH__
        return x1 + 1;
#else
        return 10;
#endif
    };
}
```

#### 17. 扩展 Lambda 的指针转换限制

```cpp
void foo(void) {
    auto lam_d = [] __device__ (double) { return 1; };
    // 错误：不能在主机代码中将扩展 lambda 转换为函数指针
    int (*fp2)(double) = lam_d;
}
```

#### 18. 扩展 Lambda 的类型特性限制

```cpp
template <typename T>
void dolaunch() {
    foo<std::is_trivially_copyable<T>::value><<<1,1>>>();
}

int main() {
    auto lam1 = [=] __host__ __device__ () { return x; };
    dolaunch<decltype(lam1)>();
}
```

与 `__device__` lambda 不同，`__host__ __device__` lambda 可以从主机代码中调用。

如前所述，CUDA 编译器会将定义在主机代码中的扩展 lambda 表达式替换为一个命名占位符类型（named placeholder type）的实例。

对于扩展的 `__host__ __device__` lambda，占位符类型会通过间接函数调用来调用原始 lambda 的 `operator()`。

间接函数调用的存在可能导致扩展的 `__host__ __device__` lambda 在主机编译器中的优化程度低于仅隐式或显式标记为 `__host__` 的 lambda。

在后一种情况下，主机编译器可以轻松地将 lambda 的主体内联到调用上下文中。

但对于扩展的 `__host__ __device__` lambda，主机编译器会遇到间接函数调用，可能无法轻松地将原始的 `__host__ __device__` lambda 主体内联。

当一个 lambda 定义在非静态类成员函数中，并且 lambda 的主体引用了类的成员变量时，C++11/C++14 规则要求通过值捕获 `this` 指针，而不是直接捕获引用的成员变量。如果 lambda 是一个定义在主机函数中的扩展 `__device__` 或 `__host__ __device__` lambda，并且该 lambda 在 GPU 上执行，那么当 `this` 指针指向主机内存时，访问引用的成员变量会导致运行时错误。

以下代码展示了在 C++11/C++14 中，lambda 捕获 `this` 指针的问题：

```cpp
#include <cstdio>

// 定义一个 CUDA 核函数，用于执行传入的可调用对象
template <typename T>
__global__ void foo(T in) { 
    printf("\n value = %d", in()); 
}

// 定义一个结构体 S1_t
struct S1_t {
    int xxx; // 成员变量

    // 构造函数，初始化成员变量 xxx
    __host__ __device__ S1_t(void) : xxx(10) { };

    // 成员函数 doit，定义了一个扩展的 __device__ lambda
    void doit(void) {
        auto lam1 = [=] __device__ {
            // 引用成员变量 xxx，导致 this 指针被捕获
            return xxx + 1;
        };

        // 启动核函数，但由于 this 指针指向主机内存，会导致运行时错误
        foo<<<1,1>>>(lam1);
        cudaDeviceSynchronize();
    }
};

int main(void) {
    S1_t s1; // 创建 S1_t 的实例
    s1.doit(); // 调用 doit 函数
}
```

C++17 引入了 `*this` 捕获模式，通过值捕获整个对象，而不是 `this` 指针。CUDA 编译器支持在 `__device__` 和 `__global__` 函数中定义的 lambda，以及在主机代码中定义的扩展 `__device__` lambda 使用 `*this` 捕获模式。

```cpp
#include <cstdio>

// 定义一个 CUDA 核函数，用于执行传入的可调用对象
template <typename T>
__global__ void foo(T in) { 
    printf("\n value = %d", in()); 
}

// 定义一个结构体 S1_t
struct S1_t {
    int xxx; // 成员变量

    // 构造函数，初始化成员变量 xxx
    __host__ __device__ S1_t(void) : xxx(10) { };

    // 成员函数 doit，定义了一个扩展的 __device__ lambda，并使用 *this 捕获模式
    void doit(void) {
        // 使用 *this 捕获模式，捕获整个对象
        auto lam1 = [=, *this] __device__ {
            // 引用成员变量 xxx，此时访问的是对象的副本
            return xxx + 1;
        };

        // 启动核函数，此时可以正确访问成员变量
        foo<<<1,1>>>(lam1);
        cudaDeviceSynchronize();
    }
};

int main(void) {
    S1_t s1; // 创建 S1_t 的实例
    s1.doit(); // 调用 doit 函数
}
```

#### `*this` 捕获模式的限制

`*this` 捕获模式不能用于未注解的 lambda（即在主机代码中定义的普通 lambda），

也不能用于扩展的 `__host__ __device__` lambda。

```cpp
struct S1_t {
    int xxx; // 成员变量

    // 构造函数，初始化成员变量 xxx
    __host__ __device__ S1_t(void) : xxx(10) { };

    // 主机函数，定义了一个扩展的 __device__ lambda，支持 *this 捕获模式
    void host_func(void) {
        auto lam1 = [=, *this] __device__ { return xxx; }; // 合法
        auto lam2 = [=, *this] __host__ __device__ { return xxx; }; // 错误：不支持扩展的 __host__ __device__ lambda
        auto lam3 = [=, *this] { return xxx; }; // 错误：不支持未注解的 lambda
    }

    // 设备函数，定义了一个 lambda，支持 *this 捕获模式
    __device__ void device_func(void) {
        auto lam1 = [=, *this] __device__ { return xxx; }; // 合法
        auto lam2 = [=, *this] __host__ __device__ { return xxx; }; // 合法
        auto lam3 = [=, *this] { return xxx; }; // 合法
    }

    // 主机-设备函数，定义了一个扩展的 __device__ lambda，支持 *this 捕获模式
    __host__ __device__ void host_device_func(void) {
        auto lam1 = [=, *this] __device__ { return xxx; }; // 合法
        auto lam2 = [=, *this] __host__ __device__ { return xxx; }; // 错误：不支持扩展的 __host__ __device__ lambda
        auto lam3 = [=, *this] { return xxx; }; // 错误：不支持未注解的 lambda
    }
};
```

## 17.8 Code Samples

以下是一个表示 RGBA 像素的类 `PixelRGBA`，并展示了如何在设备代码中使用它。

```cpp
class PixelRGBA {
public:
    // 默认构造函数，初始化 RGBA 值为 0
    __device__ PixelRGBA(): r_(0), g_(0), b_(0), a_(0) { }

    // 带参数的构造函数，初始化 RGBA 值
    __device__ PixelRGBA(unsigned char r, unsigned char g,
                         unsigned char b, unsigned char a = 255):
        r_(r), g_(g), b_(b), a_(a) { }

private:
    unsigned char r_, g_, b_, a_; // RGBA 分量
    friend PixelRGBA operator+(const PixelRGBA&, const PixelRGBA&); // 声明友元函数
};

// 重载 + 运算符，用于两个 PixelRGBA 对象的相加
__device__
PixelRGBA operator+(const PixelRGBA& p1, const PixelRGBA& p2) {
    return PixelRGBA(p1.r_ + p2.r_, p1.g_ + p2.g_,
                     p1.b_ + p2.b_, p1.a_ + p2.a_);
}

// 设备函数，演示如何使用 PixelRGBA 类
__device__ void func(void) {
    PixelRGBA p1, p2; // 创建两个 PixelRGBA 对象
    // ... 初始化 p1 和 p2
    PixelRGBA p3 = p1 + p2; // 使用重载的 + 运算符
}
```

以下是一个基类 `Shape` 和派生类 `Point` 的示例，展示了如何在设备代码中使用继承和虚函数。

```cpp
// 设备内存分配和释放函数
__device__ void* operator new(size_t bytes, MemoryPool& p);
__device__ void operator delete(void*, MemoryPool& p);

// 基类 Shape
class Shape {
public:
    __device__ Shape(void) { } // 构造函数
    __device__ void putThis(PrintBuffer *p) const; // 成员函数
    __device__ virtual void Draw(PrintBuffer *p) const { // 虚函数
        p->put("Shapeless");
    }
    __device__ virtual ~Shape() {} // 虚析构函数
};

// 派生类 Point
class Point : public Shape {
public:
    __device__ Point() : x(0), y(0) {} // 默认构造函数
    __device__ Point(int ix, int iy) : x(ix), y(iy) { } // 带参数的构造函数
    __device__ void PutCoord(PrintBuffer *p) const; // 成员函数
    __device__ void Draw(PrintBuffer *p) const; // 重写虚函数
    __device__ ~Point() {} // 析构函数
private:
    int x, y; // 坐标
};

// 设备函数，返回一个 Point 对象
__device__ Shape* GetPointObj(MemoryPool& pool) {
    Shape* shape = new(pool) Point(rand(-20,10), rand(-100,-20)); // 在内存池中分配 Point 对象
    return shape;
}
```

以下是一个类模板 `myValues` 的示例，展示了如何在设备代码中使用模板。

```cpp
// 类模板 myValues
template <class T>
class myValues {
    T values[MAX_VALUES]; // 存储值的数组
public:
    __device__ myValues(T clear) { ... } // 构造函数
    __device__ void setValue(int Idx, T value) { ... } // 设置值
    __device__ void putToMemory(T* valueLocation) { ... } // 将值写入内存
};

// 核函数模板，使用 myValues 类
template <class T>
void __global__ useValues(T* memoryBuffer) {
    myValues<T> myLocation(0); // 创建 myValues 对象
    ...
}

// 设备代码
__device__ void* buffer;

// 主机代码
int main() {
    ...
    useValues<int><<<blocks, threads>>>(buffer); // 调用核函数模板
    ...
}
```

以下是一个函数模板 `func` 的示例，展示了如何在设备代码中使用模板特化和隐式参数推导。

```cpp
// 函数模板 func
template <typename T>
__device__ bool func(T x) {
    ...
    return (...);
}

// 特化版本，针对 int 类型
template <>
__device__ bool func<int>(T x) {
    return true;
}

// 显式指定模板参数
bool result = func<double>(0.5);

// 隐式参数推导
int x = 1;
bool result = func(x);
```

以下是两个仿函数类 `Add` 和 `Sub` 的示例，展示了如何在核函数中使用仿函数。

```cpp
// 仿函数类 Add，用于加法操作
class Add {
public:
    __device__
    float operator() (float a, float b) const {
        return a + b;
    }
};

// 仿函数类 Sub，用于减法操作
class Sub {
public:
    __device__
    float operator() (float a, float b) const {
        return a - b;
    }
};

// 核函数模板，使用仿函数对两个向量进行操作
template<class O> __global__
void VectorOperation(const float * A, const float * B, float * C,
                     unsigned int N, O op) {
    unsigned int iElement = blockDim.x * blockIdx.x + threadIdx.x;
    if (iElement < N)
        C[iElement] = op(A[iElement], B[iElement]); // 使用仿函数
}

// 主机代码
int main() {
    ...
    VectorOperation<<<blocks, threads>>>(v1, v2, v3, N, Add()); // 调用核函数，使用 Add 仿函数
    ...
}
```
