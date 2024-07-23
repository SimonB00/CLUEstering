#

if [[ $1 == "-h" || $1 == "--help" ]]
then
  echo "-h, --help: Print the list of flags"
  echo "-c, --clean: Delete the output files generated during the execution"
  exit
fi

# Test the different input data types
echo "## Testing that CLUE works for all the supported data types"
python3 -m pytest test_input_datatypes.py

# Run the tests of the test datasets
echo "## Testing the algorithm on the main datasets"
python3 -m pytest test_*_dataset.py

# Test that the method choose_kernel raises an exception when used improperly
echo "## Test the exceptions of the method choose_kernel"
python3 -m pytest test_kernels.py

# Test the utility function for generating testing blobs
echo "## Test the test_blobs function"
python3 -m pytest test_test_blobs.py

# Test the equality operator for the results of the clustering
echo "## Test the equality operator for the results of the clustering"
python3 -m pytest test_clusterer_equality.py

# Test the method of changind the domain extremes of the coordinates
echo "## Test the change_domains method"
python3 -m pytest test_change_domains.py

# Test the clustering of points at the opposite extremes of a finite domain
echo "## Test the clustering of points at the opposite extremes of a finite domain"
python3 -m pytest test_domain_extremes.py

if [[ $1 == "-c" || $1 == "--clean" ]]
then
  rm -f ./*_output.csv
fi
