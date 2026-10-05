import jax.numpy as jnp

from ..operators import D, D_inv


N = 5

D1 = D(N)
D2 = D(N)

D_inv1 = D_inv(N)
D_inv2 = D_inv(N)

print("D =")
print(D1)

print("\nD_inv =")
print(D_inv1)

print("\nD(N) identique au deuxième appel :", D1 is D2)
print("D_inv(N) identique au deuxième appel :", D_inv1 is D_inv2)

x = jnp.arange(N, dtype=float)

print("\nx =", x)
print("D @ x =", D1 @ x)
print("D_inv @ x =", D_inv1 @ x)
