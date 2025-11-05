--------------------------------------------------------------------------------
-- Utilities

-- The value of the constant `pi`.
pi = 3.141592

--------------------------------------------------------------------------------
-- Inputs

-- We first define a new name for the type of inputs of the network.
-- In particular, it takes inputs of the form of a tensor of 15 real numbers:
-- 10 states and 5 reference errors 

type Input = Tensor Real [15]

-- Next we add meaningful names for the indices
-- TODO: confirm this is the correct index labels for each variable

x = 0 -- measured in metres
z = 1 -- measured in metres
theta = 2 -- measured in radians
x_dot = 3 -- measured in metres/second
z_dot = 4 -- measured in metres/second
theta_dot = 5 -- measured in radians/second
l = 6 -- measured in metres
phi = 7 -- measured in radians
l_dot = 8 -- measured in metres/second
phi_dot = 9 -- measured in radians/second
err_x = 10 -- measured in metres
err_z = 11 -- measured in metres
err_theta = 12 -- measured in radians
err_phi = 13 -- measured in radians
err_l = 14 -- measured in metres

--------------------------------------------------------------------------------
-- Outputs

-- Outputs are a tensor of 5 reals. Each one representing the score
-- for the 5 available .

type Output = Tensor Real [5]

-- Again we define meaningful names for the indices into output vectors.
-- TODO: also confirm this is correct

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
drone : Input -> Output

--------------------------------------------------------------------------------
--Normalisation 

-- The neural network operates over normalised inputs.
-- These must be scaled versions of the physical quantities
-- (position, angle, velocity, etc.) defined in real-world units.
-- The following section defines the normalisation procedure
-- so that verification can be done in the problem-space domain.

type UnnormalisedInput = Tensor Real [15]

-- Minimum and maximum problem-space values for each input dimension.
-- These define the input domain of the neural network.
-- BELOW ARE JUST PLACEHOLDERS.. NOT OUR ACTUAL STATES

minimumInputValues : UnnormalisedInput
minimumInputValues = [
  -10.0,   -- x position [m]
   0.0,    -- z (altitude) [m]
  -pi,     -- roll φ [ra
  -pi,     -- pitch θ [rad]
  -pi,     -- yaw ψ [rad]
  -10.0,   -- x velocity [m/s]
  -10.0,   -- z velocity [m/s]
  -20.0,   -- roll rate [rad/s]
  -20.0,   -- pitch rate [rad/s]
  -20.0,   -- yaw rate [rad/s]
  -2.0,    -- payload swing angle [rad]
  -5.0,    -- payload angular rate [rad/s]
   0.0,    -- cable length [m]
  -1.0,    -- cable rate [m/s]
  -1.0     -- control bias or residual term
]

maximumInputValues : UnnormalisedInput
maximumInputValues = [
   10.0,   -- x position
   10.0,   -- z altitude
    pi,    -- roll
    pi,    -- pitch
    pi,    -- yaw
   10.0,   -- x velocity
   10.0,   -- z velocity
   20.0,   -- roll rate
   20.0,   -- pitch rate
   20.0,   -- yaw rate
    2.0,   -- payload swing
    5.0,   -- payload angular rate
   2.0,    -- cable length
    1.0,   -- cable rate
    1.0    -- control bias
]
 
