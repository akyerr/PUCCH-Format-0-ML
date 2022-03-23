function pr_seq = pseudorandom_sequence(cinit, Mpn)
% 31 is the gold sequence length;

Nc = 1600; % sequence offset from initial time

x1 = zeros(Nc + sum(Mpn) + 31, 1);
x2 = x1;

% intial values x1(0) and x2(0)
x10 = 1;
% Convert to binary (str), convert string to double array, flip
x20 = flip(double((dec2bin(cinit,31) ~= '0')));

% insert initial values
x1(1) = x10;
x2(1:31) = x20;

for n = 1:(sum(Mpn) + Nc)
    x1(n + 31) = mod(x1(n + 3) + x1(n), 2);
    x2(n + 31) = mod(x2(n + 3) + x2(n + 2) + x2(n + 1) + x2(n), 2);
end
% Generate the resulting sequence (Gold Sequence) : c()
pr_seq0 = mod(x1((1:sum(Mpn)) + Nc) + x2((1:sum(Mpn)) + Nc),2);

if ~isscalar(Mpn)
    pr_seq = pr_seq0(end - Mpn(2)+1: end);
else
    pr_seq = pr_seq0;
end

end
