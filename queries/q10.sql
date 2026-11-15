-- TPC-H Q10 (returned item reporting) - supported: 4-table join + group/agg
SELECT
    c.c_custkey,
    c.c_name,
    SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
    c.c_acctbal,
    n.n_name,
    c.c_address,
    c.c_phone,
    c.c_comment
FROM customer AS c
JOIN orders AS o ON c.c_custkey = o.o_custkey
JOIN lineitem AS l ON o.o_orderkey = l.l_orderkey
JOIN nation AS n ON c.c_nationkey = n.n_nationkey
WHERE o.o_orderdate >= DATE '1993-10-01'
  AND o.o_orderdate < DATE '1994-01-01'
  AND l.l_returnflag = 'R'
GROUP BY c.c_custkey, c.c_name, c.c_acctbal, c.c_address,
         c.c_phone, c.c_comment, n.n_name
ORDER BY revenue DESC
LIMIT 20
