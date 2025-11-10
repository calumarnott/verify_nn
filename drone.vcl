--------------------------------------------------------------------------------
-- Utilities

-- The value of the constant `pi`.
pi = 3.141592

-- Absolute value function
abs : Real -> Real
abs x = if x >= 0.0 then x else -x

--------------------------------------------------------------------------------
-- Inputs

-- Inputs are a tensor of 15 reals. Each one representing a different
-- physical quantity describing the state of the drone and its payload.

type Input = Tensor Real [15]
-- Normalised input vector (each element scaled to the [-1, 1] or [0, 1] range)
-- This is what the neural network 'drone' actually receives.

-- The index mapping corresponds to the underlying physical quantities:
x_p       = 0 -- normalised x position (was metres)
z         = 1 -- normalised altitude (was metres)
theta     = 2 -- normalised pitch angle (was radians)
x_dot     = 3 -- normalised x velocity (was m/s)
z_dot     = 4 -- normalised z velocity (was m/s)
theta_dot = 5 -- normalised pitch rate (was rad/s)
l         = 6 -- normalised cable length (was metres)
phi       = 7 -- normalised payload swing angle (was radians)
l_dot     = 8 -- normalised cable rate (was m/s)
phi_dot   = 9 -- normalised swing rate (was rad/s)
err_x     = 10 -- normalised x position error (was metres)
err_z     = 11 -- normalised z position error (was metres)
err_theta = 12 -- normalised pitch error (was radians)
err_phi   = 13 -- normalised swing angle error (was radians)
err_l     = 14 -- normalised cable length error (was metres)

--------------------------------------------------------------------------------
-- Outputs

-- Outputs are a tensor of 5 reals. Each one representing the score
-- for the 5 available .

type Output = Tensor Real [5]

-- Again we define meaningful names for the indices into output vectors.
-- TODO: confirm these are still the outputs

u_x = 0
u_z = 1
u_theta = 2
u_phi = 3
u_l = 4

--------------------------------------------------------------------------------
-- The network

-- Next we use the `network` annotation to declare the name and the type of the
-- neural network we are verifying. The implementation is passed to the compiler
-- via a reference to the ONNX file at compile time.

@network
droneNN : Input -> Output

--------------------------------------------------------------------------------
--Normalisation 

-- The neural network operates over normalised inputs.
-- These must be scaled versions of the physical quantities (position, angle, velocity, etc.) defined in real-world units.
-- The following section defines the normalisation procedure
-- so that verification can be done in the problem-space domain.

type UnnormalisedInput = Tensor Real [15]

-- Minimum and maximum problem-space values for each input dimension.
-- These define the input domain of the neural network.

minimumInputValues : UnnormalisedInput
minimumInputValues = [
  -10.0,
  0.0,
  -pi,
  -10.0,
  -10.0,
  -20.0,
  0.0,
  -pi,
  -1.0,
  -20.0,
  -20.0,
  -10.0,
  -pi,
  -pi,
  -2.0 ]

maximumInputValues : UnnormalisedInput
maximumInputValues = [
  10.0,
  10.0,
  pi,
  10.0,
  10.0,
  20.0,
  2.0,
  pi,
  1.0,
  20.0,
  20.0,
  10.0,
  pi,
  pi,
  2.0 ]

-- A function that checks whether a given unnormalised input is valid
validInput : UnnormalisedInput -> Bool
validInput x = forall i . minimumInputValues ! i <= x ! i <= maximumInputValues ! i

-- A function that computes the scaling values for normalisation.
meanScalingValues : UnnormalisedInput
meanScalingValues = foreach i . (maximumInputValues ! i + minimumInputValues ! i) / 2

-- A function that normalises a given unnormalised input
normalise : UnnormalisedInput -> Input
normalise x = foreach i .
  (x ! i - meanScalingValues ! i) /
  (maximumInputValues ! i - minimumInputValues ! i)

-- Apply the trained neural network (defined elsewhere)
normDroneNN : UnnormalisedInput -> Output
normDroneNN x = droneNN (normalise x)

-- Define predicate: network advises output i for input x
advises : Index 5 -> UnnormalisedInput -> Bool
advises i x = forall j . i != j => normDroneNN x ! i < normDroneNN x ! j

--------------------------------------------------------------------------------
-- Properties
--------------------------------------------------------------------------------

-- Property 1: For all valid unnormalised inputs, each network output
--             must lie within the defined physical limits.

-- Define per-output lower and upper bounds in the problem space.
-- These correspond to actuator or command limits for your system.

minimumOutputValues : Output
minimumOutputValues = [
  -10.0,
  -10.0,
   -5.0,
  -10.0,
    0.0 ]

maximumOutputValues : Output
maximumOutputValues = [
   10.0,
   10.0,
    5.0,
   10.0,
    0.0 ]

-- Helper predicate: an output vector is within the valid range.
validOutput : Output -> Bool
validOutput y = forall i . minimumOutputValues ! i <= y ! i <= maximumOutputValues ! i

-- Property 1 statement:
-- For all valid unnormalised inputs x, the network's normalised output
-- (mapped back to output space) must be within bounds.
@property
property1 : Bool
property1 =
  forall x .
    validInput x =>
      validOutput (normDroneNN x)

--------------------------------------------------------------------------------
-- Property 2: Equilibrium / Stability Condition

-- When all state errors and rates are zero, the network should output ≈ 0.

tolerance = 0.05  -- acceptable small deviation from zero

@property
property2 : Bool 
property2 =
  forall (x : UnnormalisedInput) .
    validInput x =>
      (let zero_state = x ! err_x == 0.0 and x ! err_z == 0.0 and x ! err_theta == 0.0 and x ! err_phi == 0.0 and x ! err_l == 0.0 and x ! x_dot == 0.0 and x ! z_dot == 0.0 and x ! theta_dot == 0.0 and x ! phi_dot == 0.0 and x ! l_dot == 0.0
       in zero_state => (forall (j : Index 5) . abs (normDroneNN x ! j) <= tolerance))

--------------------------------------------------------------------------------
-- Property 3: Lipschitz / Smoothness constraint

-- Small perturbations in the input should not cause large output changes.

-- We define a Lipschitz constant K and a maximum perturbation delta.
K = 2.0           -- Lipschitz bound: output change ≤ 2 × input change
delta = 0.1       -- Small input perturbation magnitude

@property
property3 : Bool
property3 =
  forall (x1 : UnnormalisedInput) (x2 : UnnormalisedInput) .
    validInput x1 and validInput x2 and

    -- The inputs are close in all dimensions (L_inf norm ≤ delta)
    (forall i . abs (x1 ! i - x2 ! i) <= delta)
  =>
    -- Then the outputs must be close (L_inf norm ≤ K × delta)
    (forall j . abs (normDroneNN x1 ! j - normDroneNN x2 ! j) <= K * delta)