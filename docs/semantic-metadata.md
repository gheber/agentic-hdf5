== Semantic Metadata Specification

The major feature of agentic HDF5, aside from the high-level interface, is the introduction of semantic metadata into HDF data stores. This document specifies the form that this semantic metadata will take.

=== Purpose

Semantic metadata is introduced in order to achieve a few different goals.

- Prevent misreadings of data (e.g. unit errors)
- Communicate intentions associated with data (e.g. data collected with the intent of solving specific problems, or resolving certain questions)
- Collect important, hard-to-classify information associated with raw data (e.g. provenance, known error rates, context for collection, etc.)

=== Form

Each high-level HDF5 object (dataset, group, committed datatype) may have an associated HDF attribute that acts as the semantic metadata. The semantic metadata attribute's name will be `ahdf5-smd-<original object name>`.

Regular HDF5 attributes do not have their own semantic metadata. Instead, when generating or using semantic metadata for an object, the attributes attached to that object are considered as part of the object's overall context, and are described in the object's semantic metadata attribute.

For datasets, committed datatypes, and non-root groups the semantic metadata attribute will reside in the same group as original object. The root group's semantic metadata attribute will reside in the root group.

The semantic metadata attribute is a scalar HDF attribute containing a single variable-length string. This string may contain multiple lines, with individual facts or pieces of information separated by newlines.

==== "Best Guess" Semantic Metadata

For HDF files acquired externally, about which provenance details are not available, agentic HDF5 will be able to generate "best guess" semantic metadata by examining object names, datatype, file layout, etc. Best-guess semantic metadata will be flagged as such to prevent the agent from making mistakes based on assuming it to be true with certainty.

For HDF files created by the user via agentic HDF5, or about which the user has certain information, agentic HDF5 will be able to receive user-specified semantic metadata and store it as 'definite' semantic metadata, which will always be considered accurate.

Best-guess semantic metadata is distinguished by the string "BEST GUESS:" which prefixes each entry in the semantic metadata attribute. Flagging best-guess semantic metadata on a per-entry basis allows agentic-hdf5 to easily add additional information of varying reliability over time.
