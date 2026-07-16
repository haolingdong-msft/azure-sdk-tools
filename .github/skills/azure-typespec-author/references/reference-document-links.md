# Reference Document Links

Use the section that matches the requested TypeSpec change. Prefer the focused how-to for the case, and use the library reference to confirm exact decorator, model, interface, or operation-template signatures.

## API Version Evolution

- [Versioning overview](https://azure.github.io/typespec-azure/docs/howtos/versioning/01-about-versioning/): Overview of how API versioning works in TypeSpec Azure.
- [preview → preview](https://azure.github.io/typespec-azure/docs/howtos/versioning/02-preview-after-preview/): How to add a new preview version after an existing preview version.
- [preview → stable](https://azure.github.io/typespec-azure/docs/howtos/versioning/03-stable-after-preview/): How to promote a preview version to stable.
- [stable → preview](https://azure.github.io/typespec-azure/docs/howtos/versioning/04-preview-after-stable/): How to add a new preview version after a stable version.
- [stable → stable](https://azure.github.io/typespec-azure/docs/howtos/versioning/05-stable-after-stable/): How to add a new stable version after an existing stable version.
- [Evolving APIs](https://azure.github.io/typespec-azure/docs/howtos/versioning/06-evolving-apis/): How to evolve APIs across versions by adding, removing, or modifying resources, operations, properties, and spread-model members with versioning decorators.

## ARM Resources and Operations

- [ARM service setup](https://azure.github.io/typespec-azure/docs/getstarted/azure-resource-manager/step01): Define an ARM provider namespace, service metadata, common-types version, required namespaces, and the provider operations interface.
- [ARM child resources](https://azure.github.io/typespec-azure/docs/getstarted/azure-resource-manager/step03): Define parent/child resource relationships and child-resource operation interfaces.
- [ARM custom actions](https://azure.github.io/typespec-azure/docs/getstarted/azure-resource-manager/step04): Choose standard synchronous or asynchronous ARM resource-action templates and use compliant ARM response types.
- [Azure.ResourceManager reference](https://azure.github.io/typespec-azure/docs/libraries/azure-resource-manager/reference): Full reference for ARM resource decorators, resource models, extension-resource templates, check-existence operations, CRUD/list templates, action templates, and LRO response/header models.

## Long-Running Operations

- [ARM custom actions](https://azure.github.io/typespec-azure/docs/getstarted/azure-resource-manager/step04): Select `ArmResourceActionAsync` or the matching no-content template for asynchronous ARM actions; avoid custom operations unless standard templates cannot express the scenario.
- [Azure.ResourceManager reference](https://azure.github.io/typespec-azure/docs/libraries/azure-resource-manager/reference): Confirm exact signatures for asynchronous create, update, delete, action, LRO header, and operation-status templates.
- [Data-plane long-running operations](https://azure.github.io/typespec-azure/docs/howtos/azure-core/long-running-operations/): Define Azure.Core asynchronous operations and polling/final-result behavior.

## Decorators, Constraints, and Suppressions

- [Decorator syntax](https://typespec.io/docs/language-basics/decorators/): Apply decorators directly with `@` or use augment decorators with `@@` when decorating an existing declaration from another location.
- [Built-in decorators](https://typespec.io/docs/standard-library/built-in-decorators/): Reference for built-in constraints and metadata such as `@minLength`, `@maxLength`, `@pattern`, `@format`, `@doc`, and visibility decorators.
- [Directives and `#suppress`](https://typespec.io/docs/language-basics/directives/): Suppress a specific warning with its diagnostic code and a justification; compiler errors cannot be suppressed.

## Data-Plane Operations

- [Azure.Core reference](https://azure.github.io/typespec-azure/docs/libraries/azure-core/reference): Full reference for Azure.Core decorators, interfaces, operations, and models.
- [Standard resource operations](https://azure.github.io/typespec-azure/docs/libraries/azure-core/reference/interfaces): Azure.Core operation templates such as `ResourceRead`, `ResourceList`, `ResourceCreateOrUpdate`, and `ResourceDelete`.
- [Data-plane getting started](https://azure.github.io/typespec-azure/docs/getstarted/azure-core/step01): Create data-plane TypeSpec services with Azure.Core.
- [Data-plane long-running operations](https://azure.github.io/typespec-azure/docs/howtos/azure-core/long-running-operations/): Define asynchronous data-plane operations.
