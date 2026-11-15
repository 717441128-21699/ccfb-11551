-- TPC-H Q9 (product type profit measure) - supported: multi-join, LIKE, expr
SELECT
    n.n_name AS nation,
    EXTRACT(YEAR FROM o.o_orderdate) AS o_year,
    SUM(l.l_extendedprice * (1 - l.l_discount) - ps.ps_supplycost * l.l_quantity)
        AS amount
FROM part AS p
JOIN partsupp AS ps ON p.p_partkey = ps.ps_partkey
JOIN lineitem AS l ON ps.ps_partkey = l.l_partkey
    AND ps.ps_suppkey = l.l_suppkey
JOIN orders AS o ON l.l_orderkey = o.o_orderkey
JOIN supplier AS s ON l.l_suppkey = s.s_suppkey
JOIN nation AS n ON s.s_nationkey = n.n_nationkey
WHERE p.p_name LIKE '%green%'
GROUP BY n.n_name, EXTRACT(YEAR FROM o.o_orderdate)
ORDER BY n.n_name, o_year DESC
