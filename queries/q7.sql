-- TPC-H Q7 (volume shipping) - supported: multi-join + EXTRACT(year)
SELECT
    s_nation.n_name AS supp_nation,
    c_nation.n_name AS cust_nation,
    EXTRACT(YEAR FROM l.l_shipdate) AS l_year,
    SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
FROM supplier AS s
JOIN lineitem AS l ON s.s_suppkey = l.l_suppkey
JOIN orders AS o ON l.l_orderkey = o.o_orderkey
JOIN customer AS c ON o.o_custkey = c.c_custkey
JOIN nation AS s_nation ON s.s_nationkey = s_nation.n_nationkey
JOIN nation AS c_nation ON c.c_nationkey = c_nation.n_nationkey
WHERE (s_nation.n_name = 'FRANCE' AND c_nation.n_name = 'GERMANY'
       AND l.l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31')
   OR (s_nation.n_name = 'GERMANY' AND c_nation.n_name = 'FRANCE'
       AND l.l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31')
GROUP BY s_nation.n_name, c_nation.n_name, EXTRACT(YEAR FROM l.l_shipdate)
ORDER BY supp_nation, cust_nation, l_year
