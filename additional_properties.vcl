-- HERE ARE SOME ADDITIONAL PROPERTIES WE MAY WISH TO CONSIDER FOR THE DRONE CONTROLLER NEURAL NETWORK.
-- I HAVE NOT TESTED THEM YET AND THEY ARE LIKELY VERY BUGGY

-- --------------------------------------------------------------------------------
-- -- Property X: Monotonicity in altitude error (err_z)

-- -- Increasing altitude error should not decrease commanded vertical thrust.

-- @property
-- propertyX :
--   forall (x1 : UnnormalisedInput) (x2 : UnnormalisedInput) .
--     -- Both inputs must be valid
--     validInput x1 /\ validInput x2 /\

--     -- All inputs are equal except for err_z
--     (forall i . i != err_z => x1 ! i = x2 ! i) /\

--     -- The second input has a larger altitude error
--     (x2 ! err_z > x1 ! err_z)
--   =>
--     -- Then the vertical output (u_z) should not decrease
--     normDroneNN x2 ! u_z >= normDroneNN x1 ! u_z

-- --------------------------------------------------------------------------------
-- -- Property Y: Lipschitz / Smoothness constraint

-- -- Small perturbations in the input should not cause large output changes.

-- -- We define a Lipschitz constant K and a maximum perturbation delta.
-- K = 2.0           -- Lipschitz bound: output change ≤ 2 × input change
-- delta = 0.1       -- Small input perturbation magnitude

-- @property
-- propertyY :
--   forall (x1 : UnnormalisedInput) (x2 : UnnormalisedInput) .
--     -- Both inputs are valid
--     validInput x1 /\ validInput x2 /\

--     -- The inputs are close in all dimensions (L∞ norm ≤ delta)
--     (forall i . abs (x1 ! i - x2 ! i) <= delta)
--   =>
--     -- Then the outputs must be close (L∞ norm ≤ K × delta)
--     (forall j . abs (normDroneNN x1 ! j - normDroneNN x2 ! j) <= K * delta)

-- --------------------------------------------------------------------------------
-- -- Property Z: Symmetry / Sign Consistency

-- -- Mirroring horizontal position error (err_x) should mirror horizontal thrust (u_x).

-- @property
-- propertyZ :
--   forall (x : UnnormalisedInput) .
--     validInput x =>
--       let
--         -- Define a mirrored copy of the input where err_x is negated.
--         x_mirror = foreach i .
--           if i == err_x
--             then -x ! i
--           else x ! i
--       in
--         -- The horizontal thrust output should flip sign:
--         normDroneNN x_mirror ! u_x = - normDroneNN x ! u_x
--         /\
--         -- All other outputs remain the same:
--         (forall j . j != u_x => normDroneNN x_mirror ! j = normDroneNN x ! j)