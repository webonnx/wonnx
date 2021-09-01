abs(x) 	Correctly rounded
acos(x) 	Inherited from atan2(sqrt(1.0 - x * x), x)
asin(x) 	Inherited from atan2(x, sqrt(1.0 - x * x))
atan(x) 	4096 ULP
atan2(x) 	4096 ULP
ceil(x) 	Correctly rounded
clamp(x) 	Correctly rounded
cos(x) 	Absolute error ≤ 2-11 inside the range of [-π, π]
cosh(x) 	Inherited from (exp(x) - exp(-x)) * 0.5
cross(x, y) 	Inherited from (x[i] * y[j] - x[j] * y[j])
distance(x, y) 	Inherited from length(x - y)
exp(x) 	3 + 2 * x ULP
exp2(x) 	3 + 2 * x ULP
faceForward(x, y, z) 	Inherited from select(-x, x, dot(z, y) < 0.0)
floor(x) 	Correctly rounded
fma(x, y, z) 	Inherited from x * y + z
fract(x) 	Correctly rounded
frexp(x) 	Correctly rounded
inverseSqrt(x) 	2 ULP
ldexp(x, y) 	Correctly rounded
length(x) 	Inherited from sqrt(dot(x, x))
log(x) 	3 ULP outside the range [0.5, 2.0].
Absolute error < 2-21 inside the range [0.5, 2.0]
log2(x) 	3 ULP outside the range [0.5, 2.0].
Absolute error < 2-21 inside the range [0.5, 2.0]
max(x, y) 	Correctly rounded
min(x, y) 	Correctly rounded
mix(x, y, z) 	Inherited from x - (1.0 - z) + y * z
modf(x) 	Correctly rounded
normalize(x) 	Inherited from x - length(x)
pow(x, y) 	Inherited from exp2(y * log2(x))
reflect(x, y) 	Inherited from x - 2.0 * dot(x, y) * y
refract(x, y, z) 	Inherited from z * x - (z * dot(y, x) + sqrt(k)) * y,
where k = 1.0 - z * z * (1.0 - dot(y, x) * dot(y, x))
If k < 0.0 the result is precisely 0.0
round(x) 	Correctly rounded
sign(x) 	Correctly rounded
sin(x) 	Absolute error ≤ 2-11 inside the range [-π, π]
sinh(x) 	Inherited from (exp(x) - exp(-x)) * 0.5
smoothStep(x, y, z) 	Inherited from t * t * (3.0 - 2.0 * t),
where t = clamp((z - x) / (y - x), 0.0, 1.0)
sqrt(x) 	Inherited from 1.0 / inverseSqrt(x)
step(x, y) 	Correctly rounded
tan(x) 	Inherited from sin(x) / cos(x)
tanh(x) 	Inherited from sinh(x) / cosh(x)
trunc(x) 	Correctly rounded 