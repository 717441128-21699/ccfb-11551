-- TPC-H Q19 (discounted revenue) - supported: single-table multi-condition agg
SELECT
    SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
FROM lineitem AS l
WHERE (
        l.l_shipmode IN ('AIR', 'AIR REG')
    AND l.l_shipinstruct = 'DELIVER IN PERSON'
    AND (
            (l.l_quantity BETWEEN 1 AND 11)
         OR (l.l_quantity BETWEEN 10 AND 20)
         OR (l.l_quantity BETWEEN 20 AND 30)
        )
      )
