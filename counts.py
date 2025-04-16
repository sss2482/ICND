def initialize_counts(node_states):
  counts = {}
  initial_counts = {
      'S': sum(1 for state in node_states.values() if state == 'S'),
      'P': sum(1 for state in node_states.values() if state == 'P'),
      'N': sum(1 for state in node_states.values() if state == 'N'),
      'C': sum(1 for state in node_states.values() if state == 'C'),
  }
  counts = {
      'S':[initial_counts['S']],
      'P':[initial_counts['P']],
      'N':[initial_counts['N']],
      'C':[initial_counts['C']],
  }
  print("Initial state counts:", initial_counts)
  return counts