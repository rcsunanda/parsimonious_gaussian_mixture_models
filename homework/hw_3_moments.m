syms x alpha;

pdf_exponential = piecewise(x<0, 0, x>=0, alpha * exp(-alpha * x));

pdf_exponential = alpha * exp(-alpha * x);

m1_func = x * pdf_exponential;

m1 = int(m1_func, [0 inf])

pretty(m1_func)