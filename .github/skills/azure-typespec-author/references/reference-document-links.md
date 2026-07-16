# Reference Document Links

Select only the smallest set of documents that matches the user's request. Prefer task-oriented how-to pages first; use library reference pages when exact decorators, templates, or signatures are needed.

## API Version Evolution

- [Versioning overview](https://azure.github.io/typespec-azure/docs/howtos/versioning/01-about-versioning/): Overview of how API versioning works in TypeSpec Azure.
- [preview → preview](https://azure.github.io/typespec-azure/docs/howtos/versioning/02-preview-after-preview/): How to add a new preview version after an existing preview version.
- [preview → stable](https://azure.github.io/typespec-azure/docs/howtos/versioning/03-stable-after-preview/): How to promote a preview version to stable.
- [stable → preview](https://azure.github.io/typespec-azure/docs/howtos/versioning/04-preview-after-stable/): How to add a new preview version after a stable version.
- [stable → stable](https://azure.github.io/typespec-azure/docs/howtos/versioning/05-stable-after-stable/): How to add a new stable version after an existing stable version.
- [Evolving APIs](https://azure.github.io/typespec-azure/docs/howtos/versioning/06-evolving-apis/): How to evolve your API across versions by adding, removing, or modifying resources, operations, and properties using versioning decorators.

## ARM Resource Definitions

- [Defining ARM resources](https://azure.github.io/typespec-azure/docs/getstarted/azure-resource-manager/step02/): Tutorial for defining resource properties, tracked/proxy/extension resource models, names, and standard operations.
- [ARM resource types](https://azure.github.io/typespec-azure/docs/howtos/arm/resource-type/): Choose and model tracked, proxy, tenant, extension, child, subscription, location, and singleton resources.
- [Defining child resources](https://azure.github.io/typespec-azure/docs/getstarted/azure-resource-manager/step03/): Tutorial for parent/child relationships using `@parentResource` and parent-scoped operations.
- [ARM Resource Manager data types](https://azure.github.io/typespec-azure/docs/libraries/azure-resource-manager/reference/data-types/): Exact signatures for ARM resource base types, resource parameters, and supporting models.

## ARM Resource Operations

- [ARM resource operations](https://azure.github.io/typespec-azure/docs/howtos/arm/resource-operations/): Recommended and required CRUDL operations, PATCH choices, synchronous/asynchronous variants, list scopes, actions, and property visibility.
- [Defining custom actions](https://azure.github.io/typespec-azure/docs/getstarted/azure-resource-manager/step04/): Tutorial for synchronous and asynchronous ARM resource actions, custom operations, response types, and name-availability operations.
- [ARM operation interfaces](https://azure.github.io/typespec-azure/docs/libraries/azure-resource-manager/reference/interfaces/): Exact signatures for standard, extension-resource, and provider operation interfaces.
- [ARM decorators](https://azure.github.io/typespec-azure/docs/libraries/azure-resource-manager/reference/decorators/): Exact targets and parameters for `@armResourceOperations`, resource lifecycle/action decorators, provider decorators, and scope decorators.
- [Deep Dive: Long-running (Asynchronous) Operations](https://azure.github.io/typespec-azure/docs/howtos/azure-core/long-running-operations/): LRO polling, status monitors, operation/resource links, and custom LRO patterns.

## Data-Plane Operations

- [Azure.Core reference](https://azure.github.io/typespec-azure/docs/libraries/azure-core/reference): Full reference for Azure.Core decorators, interfaces, operations, and models.
- [Standard resource operations](https://azure.github.io/typespec-azure/docs/libraries/azure-core/reference/interfaces): Azure.Core operation templates (`ResourceRead`, `ResourceList`, `ResourceCreateOrUpdate`, `ResourceDelete`, etc.).
- [Data-plane getting started](https://azure.github.io/typespec-azure/docs/getstarted/azure-core/step01): Getting started guide for creating data-plane TypeSpec services with Azure.Core.
- [Custom resource actions](https://azure.github.io/typespec-azure/docs/getstarted/azure-core/step07/): Define instance and collection actions with Azure.Core operation signatures.
- [Customizing operations with traits](https://azure.github.io/typespec-azure/docs/getstarted/azure-core/step08/): Customize standard operation parameters and response bodies through traits.

## Validation, Warnings, and Style

- [ARM rules, linting, and suppression](https://azure.github.io/typespec-azure/docs/howtos/arm/arm-rules/): Diagnose ARM linter warnings, apply the recommended fix, and suppress only justified false positives or approved exceptions.
- [Azure TypeSpec Style Guide](https://azure.github.io/typespec-azure/docs/reference/azure-style-guide/): Cross-cutting Azure requirements for libraries, versioning, decorators, operation groups, naming, pagination, and authentication.
- [TypeSpec Authoring Skill sample tasks](https://azure.github.io/typespec-azure/docs/getstarted/typespec-authoring-skill/): Representative authoring requests for versioning, resource definitions, operations, models, and types.
